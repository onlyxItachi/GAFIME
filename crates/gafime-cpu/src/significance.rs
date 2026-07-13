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
//! The interaction *signal* (centered-product) is independent of `y`, so a
//! permutation only re-runs the metric reduction against a shuffled target — no
//! plan rebuild. Determinism comes from a seeded splitmix64 stream per
//! permutation/repeat, so results are reproducible and parallel-safe.

use std::collections::HashSet;

use rayon::prelude::*;

use gafime_orchestrator::{
    plan::combos::select_adaptive_mi_bins_for_backend, CompiledPlan, OrchestratorError,
    OrchestratorResult,
};
use gafime_types::{BackendKind, GafimeArityChunk, GAFIME_BACKEND_CPU, GAFIME_FAMILY_CONTINUOUS};

use crate::kernels::{self, MetricKernel};
use crate::matrix::CpuMatrix;
use crate::simd;

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

/// Build the interaction signal and target on a bootstrap resample of the rows.
/// Column means are recomputed on the resample (a faithful re-run, not a reuse of
/// the full-data mean).
fn resampled_signal_and_target(
    matrix: &CpuMatrix,
    combo: &[u32],
    indices: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let n = indices.len();
    let target = matrix.target();
    let y: Vec<f32> = indices.iter().map(|&i| target[i]).collect();

    if combo.len() == 1 {
        let col = matrix.column(combo[0] as usize);
        let x: Vec<f32> = indices.iter().map(|&i| col[i]).collect();
        return (x, y);
    }

    let mut signal = vec![1.0f32; n];
    for &feature in combo {
        let col = matrix.column(feature as usize);
        let gathered: Vec<f32> = indices.iter().map(|&i| col[i]).collect();
        let mean = if n == 0 {
            0.0
        } else {
            (gathered.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32
        };
        for (product, &value) in signal.iter_mut().zip(&gathered) {
            *product *= value - mean;
        }
    }
    (signal, y)
}

/// Score one signal against a target for every metric in `metrics`.
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

#[derive(Clone, Copy)]
enum NullFamily<'a> {
    Selected(&'a [Vec<u32>]),
    FlatPlan {
        combo_indices: &'a [u32],
        chunks: &'a [GafimeArityChunk],
    },
}

impl NullFamily<'_> {
    fn for_each_combo(self, mut visit: impl FnMut(&[u32])) {
        match self {
            Self::Selected(combos) => {
                for combo in combos {
                    visit(combo);
                }
            }
            Self::FlatPlan {
                combo_indices,
                chunks,
            } => {
                for chunk in chunks {
                    let arity = usize::try_from(chunk.arity)
                        .expect("validated null-family arity fits usize");
                    let start = usize::try_from(chunk.descriptor_offset)
                        .expect("validated null-family offset fits usize");
                    let count = usize::try_from(chunk.combo_count)
                        .expect("validated null-family count fits usize");
                    let descriptor_len = count
                        .checked_mul(arity)
                        .expect("validated null-family descriptor length");
                    let end = start
                        .checked_add(descriptor_len)
                        .expect("validated null-family descriptor range");
                    for combo in combo_indices[start..end].chunks_exact(arity) {
                        visit(combo);
                    }
                }
            }
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

    let combo_indices = plan.combo_indices();
    for chunk in plan.chunks() {
        if chunk.family != GAFIME_FAMILY_CONTINUOUS {
            return Err(OrchestratorError::InvalidPlan(
                "significance null family must be continuous",
            ));
        }
        let arity = usize::try_from(chunk.arity).map_err(|_| {
            OrchestratorError::InvalidPlan("significance null-family arity exceeds usize")
        })?;
        let start = usize::try_from(chunk.descriptor_offset).map_err(|_| {
            OrchestratorError::InvalidPlan("significance null-family offset exceeds usize")
        })?;
        let count = usize::try_from(chunk.combo_count).map_err(|_| {
            OrchestratorError::InvalidPlan("significance null-family count exceeds usize")
        })?;
        let descriptor_len = count
            .checked_mul(arity)
            .ok_or(OrchestratorError::InvalidPlan(
                "significance null-family descriptor length overflows",
            ))?;
        let end = start
            .checked_add(descriptor_len)
            .ok_or(OrchestratorError::InvalidPlan(
                "significance null-family descriptor range overflows",
            ))?;
        if end > combo_indices.len() {
            return Err(OrchestratorError::InvalidPlan(
                "significance null-family chunk exceeds descriptors",
            ));
        }
    }
    Ok(())
}

fn validate_selected_rows(
    matrix: &CpuMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[Vec<f32>],
    null_family: NullFamily<'_>,
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

/// Evaluate permutation p-values + bootstrap stability for a set of candidate
/// `combos` whose observed metric values are `observed` (aligned to `combos`,
/// each inner slice aligned to `metrics`). Returns one `CandidateSignificance`
/// per combo, in the same order. This compatibility entrypoint uses the selected
/// combos as the permutation family; callers that screened a larger plan must use
/// [`evaluate_with_null_family`] for valid family-wise correction.
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
        NullFamily::Selected(combos),
        metrics,
        params,
    )
}

/// Evaluate selected report rows while deriving each permutation's maxT null
/// statistic from every candidate in `null_family`. The plan stays in compact,
/// borrowed descriptor form: permutations stream its chunks and retain only one
/// maximum per metric. Bootstrap stability remains selected-only.
pub fn evaluate_with_null_family(
    matrix: &CpuMatrix,
    selected_combos: &[Vec<u32>],
    selected_observed: &[Vec<f32>],
    null_family: &CompiledPlan,
    metrics: &[MetricKernel],
    params: &SignificanceParams,
) -> OrchestratorResult<Vec<CandidateSignificance>> {
    validate_null_family_plan(matrix, null_family, params.backend_kind)?;
    let null_family = NullFamily::FlatPlan {
        combo_indices: null_family.combo_indices(),
        chunks: null_family.chunks(),
    };
    validate_selected_rows(
        matrix,
        selected_combos,
        selected_observed,
        null_family,
        metrics,
    )?;
    Ok(evaluate_impl(
        matrix,
        selected_combos,
        selected_observed,
        null_family,
        metrics,
        params,
    ))
}

fn evaluate_impl(
    matrix: &CpuMatrix,
    combos: &[Vec<u32>],
    observed: &[Vec<f32>],
    null_family: NullFamily<'_>,
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
    const EXCEEDANCE_EPS: f32 = 1e-6;
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
                let seed = mix_seed(params.random_seed, 0xA5A5_A5A5, p as u64);
                let shuffled = shuffled_target(target, seed);
                let mut perm_max = vec![f32::NEG_INFINITY; metric_count];
                let mut interaction_scratch = Vec::with_capacity(matrix.rows() as usize);
                let mut score_scratch = Vec::with_capacity(metric_count);
                null_family.for_each_combo(|combo| {
                    if combo.len() == 1 {
                        score_signal_into(
                            matrix.column(combo[0] as usize),
                            &shuffled,
                            metrics,
                            mi_bins,
                            mi_approximate,
                            &mut score_scratch,
                        );
                    } else {
                        interaction_signal_into(matrix, combo, &mut interaction_scratch);
                        score_signal_into(
                            &interaction_scratch,
                            &shuffled,
                            metrics,
                            mi_bins,
                            mi_approximate,
                            &mut score_scratch,
                        );
                    }
                    for (mi, &value) in score_scratch.iter().enumerate() {
                        let strength = extremeness(value, metrics[mi]);
                        if strength > perm_max[mi] {
                            perm_max[mi] = strength;
                        }
                    }
                });
                let mut local = vec![0u32; candidate_count * metric_count];
                for ci in 0..candidate_count {
                    for mi in 0..metric_count {
                        if perm_max[mi] + EXCEEDANCE_EPS >= observed_ext[ci * metric_count + mi] {
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
    let repeats = params.num_repeats.max(1) as usize;
    let rows = matrix.rows() as usize;
    let repeat_grids: Vec<Vec<f32>> = (0..repeats)
        .into_par_iter()
        .map(|r| {
            let seed = mix_seed(params.random_seed, 0x5A5A_5A5A, r as u64);
            let indices = bootstrap_indices(rows, seed);
            let mut flat = vec![0.0f32; candidate_count * metric_count];
            for (ci, combo) in combos.iter().enumerate() {
                let (signal, y) = resampled_signal_and_target(matrix, combo, &indices);
                let scores = score_signal(&signal, &y, metrics, mi_bins, mi_approximate);
                for (mi, &value) in scores.iter().enumerate() {
                    flat[ci * metric_count + mi] = value;
                }
            }
            flat
        })
        .collect();

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
                let samples: Vec<f32> = repeat_grids
                    .iter()
                    .map(|grid| grid[ci * metric_count + mi])
                    .collect();
                let (mean, std) = mean_std(&samples);
                means[mi] = mean;
                stds[mi] = std;
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
    use gafime_orchestrator::plan::combos::{build_continuous_plan, ContinuousPlanRequest};
    use gafime_types::{
        GafimeRankSpec, GAFIME_BACKEND_METAL, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    };

    fn pearson_r2() -> Vec<MetricKernel> {
        vec![MetricKernel::Pearson, MetricKernel::R2]
    }

    #[test]
    fn splitmix_shuffle_is_deterministic() {
        let y = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(shuffled_target(&y, 42), shuffled_target(&y, 42));
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
        let family = NullFamily::FlatPlan {
            combo_indices: plan.combo_indices(),
            chunks: plan.chunks(),
        };
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
