//! Private Issue #73 candidate-bank experiment.  This is not a shipping API.

use gafime_cpu::{kernels::precision::pearson_mixed, simd::pearson_sums};
use rayon::prelude::*;
use std::{collections::HashSet, mem::size_of};

const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_BANK_BYTES: usize = 64 * 1024 * 1024;
const MAX_CANDIDATES: usize = 4_096;
const PLANTED_TRAIN_ROWS: usize = 256;
const PLANTED_HOLDOUT_ROWS: usize = 192;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CandidateOp {
    Identity { column: usize },
    AbsoluteDifference { left: usize, right: usize },
    Softsign { column: usize },
    CenteredProduct2 { left: usize, right: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CandidateCatalog {
    feature_count: usize,
    candidates: Vec<CandidateOp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateError {
    Dimensions {
        rows: usize,
        columns: usize,
    },
    Shape {
        expected: usize,
        found: usize,
    },
    Bytes {
        what: &'static str,
        bytes: usize,
    },
    SizeOverflow,
    EmptyCatalog,
    CatalogTooLarge {
        found: usize,
    },
    InvalidCatalog {
        candidate: usize,
        reason: &'static str,
    },
    ColumnOutOfBounds {
        candidate: usize,
        column: usize,
        feature_count: usize,
    },
    FeatureCount {
        expected: usize,
        found: usize,
    },
    NonFiniteInput {
        row: usize,
        column: usize,
    },
    PointwiseNonFinite {
        candidate: usize,
        row: usize,
    },
    BankShape,
    IncompatibleBanks,
    MissingHoldoutTarget,
    NonFiniteTarget {
        row: usize,
    },
    ScoreUnavailable {
        candidate: usize,
    },
}
impl CandidateCatalog {
    pub(crate) fn new(
        feature_count: usize,
        candidates: Vec<CandidateOp>,
    ) -> Result<Self, CandidateError> {
        if feature_count == 0 {
            return Err(CandidateError::Dimensions {
                rows: 0,
                columns: 0,
            });
        }
        if candidates.is_empty() {
            return Err(CandidateError::EmptyCatalog);
        }
        if candidates.len() > MAX_CANDIDATES {
            return Err(CandidateError::CatalogTooLarge {
                found: candidates.len(),
            });
        }
        let mut unique = HashSet::with_capacity(candidates.len());
        for (candidate, &op) in candidates.iter().enumerate() {
            validate_op(candidate, op, feature_count)?;
            if !unique.insert(op) {
                return Err(CandidateError::InvalidCatalog {
                    candidate,
                    reason: "duplicate candidate",
                });
            }
        }
        Ok(Self {
            feature_count,
            candidates,
        })
    }
}

fn reference_catalog() -> Result<CandidateCatalog, CandidateError> {
    CandidateCatalog::new(
        2,
        vec![
            CandidateOp::Identity { column: 0 },
            CandidateOp::AbsoluteDifference { left: 0, right: 1 },
            CandidateOp::Softsign { column: 0 },
            CandidateOp::CenteredProduct2 { left: 0, right: 1 },
        ],
    )
}

fn validate_op(
    candidate: usize,
    op: CandidateOp,
    feature_count: usize,
) -> Result<(), CandidateError> {
    let check = |column| {
        if column < feature_count {
            Ok(())
        } else {
            Err(CandidateError::ColumnOutOfBounds {
                candidate,
                column,
                feature_count,
            })
        }
    };
    match op {
        CandidateOp::Identity { column } | CandidateOp::Softsign { column } => check(column),
        CandidateOp::AbsoluteDifference { left, right } => {
            if left >= right {
                return Err(CandidateError::InvalidCatalog {
                    candidate,
                    reason: "absolute difference must use left < right",
                });
            }
            check(left)?;
            check(right)
        }
        CandidateOp::CenteredProduct2 { left, right } => {
            if left >= right {
                return Err(CandidateError::InvalidCatalog {
                    candidate,
                    reason: "centered product must use left < right",
                });
            }
            check(left)?;
            check(right)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeMatrixRef<'a> {
    rows: usize,
    columns: usize,
    values_row_major: &'a [f32],
}

impl<'a> NativeMatrixRef<'a> {
    pub(crate) fn new(
        rows: usize,
        columns: usize,
        values_row_major: &'a [f32],
    ) -> Result<Self, CandidateError> {
        let matrix = Self {
            rows,
            columns,
            values_row_major,
        };
        validate_matrix(matrix)?;
        Ok(matrix)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CandidateBank {
    pub(crate) rows: usize,
    pub(crate) candidate_count: usize,
    pub(crate) values_row_major: Vec<f32>,
    pub(crate) candidate_major_bytes: usize,
    pub(crate) row_major_bytes: usize,
    /// f32 payload only: candidate-major outputs + row-major bank + worker scratch upper bound.
    pub(crate) conservative_peak_value_bytes: usize,
}

impl CandidateBank {
    pub(crate) fn row(&self, row: usize) -> Option<&[f32]> {
        let start = row.checked_mul(self.candidate_count)?;
        let end = start.checked_add(self.candidate_count)?;
        self.values_row_major.get(start..end)
    }
}

fn checked_elements(rows: usize, columns: usize) -> Result<usize, CandidateError> {
    rows.checked_mul(columns)
        .ok_or(CandidateError::SizeOverflow)
}

fn checked_f32_bytes(
    rows: usize,
    columns: usize,
    limit: usize,
    what: &'static str,
) -> Result<usize, CandidateError> {
    let bytes = checked_elements(rows, columns)?
        .checked_mul(size_of::<f32>())
        .ok_or(CandidateError::SizeOverflow)?;
    if bytes > limit {
        return Err(CandidateError::Bytes { what, bytes });
    }
    Ok(bytes)
}

fn validate_matrix(matrix: NativeMatrixRef<'_>) -> Result<(), CandidateError> {
    if matrix.rows < 2 || matrix.columns == 0 {
        return Err(CandidateError::Dimensions {
            rows: matrix.rows,
            columns: matrix.columns,
        });
    }
    let expected = checked_elements(matrix.rows, matrix.columns)?;
    if matrix.values_row_major.len() != expected {
        return Err(CandidateError::Shape {
            expected,
            found: matrix.values_row_major.len(),
        });
    }
    checked_f32_bytes(matrix.rows, matrix.columns, MAX_INPUT_BYTES, "input")?;
    for (index, &value) in matrix.values_row_major.iter().enumerate() {
        if !value.is_finite() {
            return Err(CandidateError::NonFiniteInput {
                row: index / matrix.columns,
                column: index % matrix.columns,
            });
        }
    }
    Ok(())
}

/// Deterministic finite f32 input for row/candidate sweep harnesses.
pub(crate) fn synthetic_inputs(
    rows: usize,
    features: usize,
    seed: u64,
) -> Result<Vec<f32>, CandidateError> {
    if rows < 2 || features == 0 {
        return Err(CandidateError::Dimensions {
            rows,
            columns: features,
        });
    }
    let elements = checked_elements(rows, features)?;
    checked_f32_bytes(rows, features, MAX_INPUT_BYTES, "input")?;
    let mut rng = SplitMix64::new(seed);
    Ok((0..elements).map(|_| rng.signed()).collect())
}

#[derive(Default)]
struct MaterializeScratch {
    values: Vec<f32>,
}

/// Materialize native f32 transforms once.  The resulting row-major bank can be
/// treated as ordinary unary features by either existing Core or CUDA scorers.
pub(crate) fn materialize_row_major(
    input: &NativeMatrixRef<'_>,
    catalog: &CandidateCatalog,
) -> Result<CandidateBank, CandidateError> {
    validate_matrix(*input)?;
    if input.columns != catalog.feature_count {
        return Err(CandidateError::FeatureCount {
            expected: catalog.feature_count,
            found: input.columns,
        });
    }
    let bank_bytes = checked_f32_bytes(
        input.rows,
        catalog.candidates.len(),
        MAX_BANK_BYTES,
        "candidate bank",
    )?;
    let means = mixed_f32_means(*input);
    let candidate_major = catalog
        .candidates
        .par_iter()
        .copied()
        .enumerate()
        .map_init(MaterializeScratch::default, |scratch, (candidate, op)| {
            materialize_one(*input, &means, candidate, op, scratch)
        })
        .collect::<Result<Vec<_>, CandidateError>>()?;

    let mut values_row_major = vec![0.0; checked_elements(input.rows, catalog.candidates.len())?];
    for (candidate, values) in candidate_major.iter().enumerate() {
        for (row, &value) in values.iter().enumerate() {
            values_row_major[row * catalog.candidates.len() + candidate] = value;
        }
    }
    Ok(CandidateBank {
        rows: input.rows,
        candidate_count: catalog.candidates.len(),
        values_row_major,
        candidate_major_bytes: bank_bytes,
        row_major_bytes: bank_bytes,
        conservative_peak_value_bytes: bank_bytes
            .checked_mul(3)
            .ok_or(CandidateError::SizeOverflow)?,
    })
}

fn mixed_f32_means(input: NativeMatrixRef<'_>) -> Vec<f32> {
    let mut sums = vec![0.0f64; input.columns];
    for row in input.values_row_major.chunks_exact(input.columns) {
        for (sum, &value) in sums.iter_mut().zip(row) {
            *sum += value as f64;
        }
    }
    sums.into_iter()
        .map(|sum| (sum / input.rows as f64) as f32)
        .collect()
}

fn materialize_one(
    input: NativeMatrixRef<'_>,
    means: &[f32],
    candidate: usize,
    op: CandidateOp,
    scratch: &mut MaterializeScratch,
) -> Result<Vec<f32>, CandidateError> {
    scratch.values.clear();
    scratch.values.reserve(input.rows);
    for row in 0..input.rows {
        let value = |column| input.values_row_major[row * input.columns + column];
        let output = match op {
            CandidateOp::Identity { column } => value(column),
            CandidateOp::AbsoluteDifference { left, right } => (value(left) - value(right)).abs(),
            CandidateOp::Softsign { column } => {
                let value = value(column);
                value / (1.0 + value.abs())
            }
            CandidateOp::CenteredProduct2 { left, right } => {
                (value(left) - means[left]) * (value(right) - means[right])
            }
        };
        if !output.is_finite() {
            return Err(CandidateError::PointwiseNonFinite { candidate, row });
        }
        scratch.values.push(output);
    }
    // Keep the worker-local allocation for later candidates; cloning is the
    // candidate-major temporary accounted for in `conservative_peak_value_bytes`.
    Ok(scratch.values.clone())
}

#[derive(Default)]
struct PairScratch {
    left: Vec<f32>,
    right: Vec<f32>,
}

/// Raw, ordered paired-view Pearson values; no label is accepted here.
pub(crate) fn paired_view_scores(
    primary: &CandidateBank,
    alternate: &CandidateBank,
) -> Result<Vec<f64>, CandidateError> {
    validate_bank(primary)?;
    validate_bank(alternate)?;
    if primary.rows != alternate.rows || primary.candidate_count != alternate.candidate_count {
        return Err(CandidateError::IncompatibleBanks);
    }
    (0..primary.candidate_count)
        .into_par_iter()
        .map_init(PairScratch::default, |scratch, candidate| {
            scratch.left.clear();
            scratch.right.clear();
            scratch.left.reserve(primary.rows);
            scratch.right.reserve(primary.rows);
            for row in 0..primary.rows {
                scratch
                    .left
                    .push(primary.values_row_major[row * primary.candidate_count + candidate]);
                scratch
                    .right
                    .push(alternate.values_row_major[row * alternate.candidate_count + candidate]);
            }
            finite_pearson(&scratch.left, &scratch.right, candidate)
        })
        .collect::<Result<Vec<_>, CandidateError>>()
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewSelection {
    pub(crate) selected_index: usize,
    pub(crate) per_candidate_scores: Vec<f64>,
}

/// Deterministic first-index tie break over only the training paired views.
pub(crate) fn select_train_view_consistency(
    primary: &CandidateBank,
    alternate: &CandidateBank,
) -> Result<ViewSelection, CandidateError> {
    let per_candidate_scores = paired_view_scores(primary, alternate)?;
    let mut selected_index = 0;
    for index in 1..per_candidate_scores.len() {
        if per_candidate_scores[index]
            .total_cmp(&per_candidate_scores[selected_index])
            .is_gt()
        {
            selected_index = index;
        }
    }
    Ok(ViewSelection {
        selected_index,
        per_candidate_scores,
    })
}

/// Holdout-only raw target correlations, deliberately separate from discovery.
pub(crate) fn holdout_target_scores(
    bank: &CandidateBank,
    target: &[f32],
) -> Result<Vec<f64>, CandidateError> {
    validate_bank(bank)?;
    if target.len() != bank.rows {
        return Err(CandidateError::Shape {
            expected: bank.rows,
            found: target.len(),
        });
    }
    if let Some((row, _)) = target
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CandidateError::NonFiniteTarget { row });
    }
    (0..bank.candidate_count)
        .into_par_iter()
        .map_init(PairScratch::default, |scratch, candidate| {
            scratch.left.clear();
            scratch.right.clear();
            scratch.left.reserve(bank.rows);
            scratch.right.reserve(bank.rows);
            for (row, &target_value) in target.iter().enumerate() {
                scratch
                    .left
                    .push(bank.values_row_major[row * bank.candidate_count + candidate]);
                scratch.right.push(target_value);
            }
            finite_pearson(&scratch.left, &scratch.right, candidate)
        })
        .collect::<Result<Vec<_>, CandidateError>>()
}

fn validate_bank(bank: &CandidateBank) -> Result<(), CandidateError> {
    if bank.rows < 2 || bank.candidate_count == 0 {
        return Err(CandidateError::BankShape);
    }
    let expected = checked_elements(bank.rows, bank.candidate_count)?;
    if bank.values_row_major.len() != expected {
        return Err(CandidateError::BankShape);
    }
    checked_f32_bytes(
        bank.rows,
        bank.candidate_count,
        MAX_BANK_BYTES,
        "candidate bank",
    )?;
    if bank.values_row_major.iter().any(|value| !value.is_finite()) {
        return Err(CandidateError::BankShape);
    }
    Ok(())
}

fn finite_pearson(left: &[f32], right: &[f32], candidate: usize) -> Result<f64, CandidateError> {
    let sums = pearson_sums(left, right);
    if sums.n != left.len()
        || !sums.sx.is_finite()
        || !sums.sy.is_finite()
        || !sums.sxx.is_finite()
        || !sums.syy.is_finite()
        || !sums.sxy.is_finite()
        || sums.sxx <= 0.0
        || sums.syy <= 0.0
    {
        return Err(CandidateError::ScoreUnavailable { candidate });
    }
    let score = pearson_mixed(left, right);
    if score.is_finite() {
        Ok(score)
    } else {
        Err(CandidateError::ScoreUnavailable { candidate })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PlantedPairedViewFixture {
    pub(crate) catalog: CandidateCatalog,
    pub(crate) train_primary: Vec<f32>,
    pub(crate) train_aligned: Vec<f32>,
    pub(crate) train_shuffled: Vec<f32>,
    pub(crate) holdout_primary: Vec<f32>,
    pub(crate) holdout_target: Vec<f32>,
    pub(crate) train_rows: usize,
    pub(crate) holdout_rows: usize,
    pub(crate) feature_count: usize,
}

/// Identity-control fixture: discovery has paired views only, never y.
pub(crate) fn planted_fixture(seed: u64) -> Result<PlantedPairedViewFixture, CandidateError> {
    let feature_count = 2;
    let (train_primary, train_aligned) = planted_train(
        PLANTED_TRAIN_ROWS,
        seeds(seed, 1),
        seeds(seed, 2),
        seeds(seed, 3),
        seeds(seed, 4),
        seeds(seed, 5),
    );
    let train_shuffled = shuffled_rows(
        PLANTED_TRAIN_ROWS,
        feature_count,
        &train_aligned,
        seeds(seed, 6),
    );
    let (holdout_primary, holdout_target) = planted_holdout(
        PLANTED_HOLDOUT_ROWS,
        seeds(seed, 7),
        seeds(seed, 8),
        seeds(seed, 9),
        seeds(seed, 10),
    );
    Ok(PlantedPairedViewFixture {
        catalog: reference_catalog()?,
        train_primary,
        train_aligned,
        train_shuffled,
        holdout_primary,
        holdout_target,
        train_rows: PLANTED_TRAIN_ROWS,
        holdout_rows: PLANTED_HOLDOUT_ROWS,
        feature_count,
    })
}

/// Invariance-positive fixture: only abs-difference removes shared nuisance.
pub(crate) fn planted_invariance_fixture(
    seed: u64,
) -> Result<PlantedPairedViewFixture, CandidateError> {
    let feature_count = 2;
    let (train_primary, train_aligned, _) = invariance_rows(
        PLANTED_TRAIN_ROWS,
        seeds(seed, 21),
        seeds(seed, 22),
        seeds(seed, 23),
        seeds(seed, 24),
        seeds(seed, 25),
        None,
    );
    let train_shuffled = shuffled_rows(
        PLANTED_TRAIN_ROWS,
        feature_count,
        &train_aligned,
        seeds(seed, 26),
    );
    let (holdout_primary, _, holdout_target) = invariance_rows(
        PLANTED_HOLDOUT_ROWS,
        seeds(seed, 31),
        seeds(seed, 32),
        seeds(seed, 33),
        seeds(seed, 34),
        seeds(seed, 35),
        Some(seeds(seed, 36)),
    );
    Ok(PlantedPairedViewFixture {
        catalog: reference_catalog()?,
        train_primary,
        train_aligned,
        train_shuffled,
        holdout_primary,
        holdout_target: holdout_target.ok_or(CandidateError::MissingHoldoutTarget)?,
        train_rows: PLANTED_TRAIN_ROWS,
        holdout_rows: PLANTED_HOLDOUT_ROWS,
        feature_count,
    })
}

fn planted_train(
    rows: usize,
    latent_seed: u64,
    primary_noise_seed: u64,
    alternate_noise_seed: u64,
    primary_nuisance_seed: u64,
    alternate_nuisance_seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut latent = SplitMix64::new(latent_seed);
    let mut primary_noise = SplitMix64::new(primary_noise_seed);
    let mut alternate_noise = SplitMix64::new(alternate_noise_seed);
    let mut primary_nuisance = SplitMix64::new(primary_nuisance_seed);
    let mut alternate_nuisance = SplitMix64::new(alternate_nuisance_seed);
    let mut primary = Vec::with_capacity(rows * 2);
    let mut alternate = Vec::with_capacity(rows * 2);
    for _ in 0..rows {
        let signal = latent.signed() * 1.5;
        primary.extend([
            signal + primary_noise.signed() * 0.025,
            primary_nuisance.signed() * 1.5,
        ]);
        alternate.extend([
            signal + alternate_noise.signed() * 0.025,
            alternate_nuisance.signed() * 1.5,
        ]);
    }
    (primary, alternate)
}

fn planted_holdout(
    rows: usize,
    latent_seed: u64,
    feature_noise_seed: u64,
    nuisance_seed: u64,
    target_noise_seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut latent = SplitMix64::new(latent_seed);
    let mut feature_noise = SplitMix64::new(feature_noise_seed);
    let mut nuisance = SplitMix64::new(nuisance_seed);
    let mut target_noise = SplitMix64::new(target_noise_seed);
    let mut primary = Vec::with_capacity(rows * 2);
    let mut target = Vec::with_capacity(rows);
    for _ in 0..rows {
        let signal = latent.signed() * 1.5;
        primary.extend([
            signal + feature_noise.signed() * 0.025,
            nuisance.signed() * 1.5,
        ]);
        target.push(signal + target_noise.signed() * 0.04);
    }
    (primary, target)
}

fn invariance_rows(
    rows: usize,
    magnitude_seed: u64,
    primary_sign_seed: u64,
    alternate_sign_seed: u64,
    primary_nuisance_seed: u64,
    alternate_nuisance_seed: u64,
    target_noise_seed: Option<u64>,
) -> (Vec<f32>, Vec<f32>, Option<Vec<f32>>) {
    let mut magnitude_rng = SplitMix64::new(magnitude_seed);
    let mut primary_sign = SplitMix64::new(primary_sign_seed);
    let mut alternate_sign = SplitMix64::new(alternate_sign_seed);
    let mut primary_nuisance = SplitMix64::new(primary_nuisance_seed);
    let mut alternate_nuisance = SplitMix64::new(alternate_nuisance_seed);
    let mut target_noise = target_noise_seed.map(SplitMix64::new);
    let mut primary = Vec::with_capacity(rows * 2);
    let mut alternate = Vec::with_capacity(rows * 2);
    let mut target = target_noise_seed.map(|_| Vec::with_capacity(rows));
    for _ in 0..rows {
        let magnitude = 0.2 + magnitude_rng.signed().abs() * 0.8;
        let primary_signal = if primary_sign.next_u64() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        let alternate_signal = if alternate_sign.next_u64() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        let nuisance = 1.0 + primary_nuisance.signed() * 0.25;
        let alternate_nuisance_value = 1.0 + alternate_nuisance.signed() * 0.25;
        primary.extend([nuisance + primary_signal, nuisance]);
        alternate.extend([
            alternate_nuisance_value + alternate_signal,
            alternate_nuisance_value,
        ]);
        if let (Some(target), Some(noise)) = (target.as_mut(), target_noise.as_mut()) {
            target.push(magnitude + noise.signed() * 0.02);
        }
    }
    (primary, alternate, target)
}

fn shuffled_rows(rows: usize, columns: usize, values: &[f32], seed: u64) -> Vec<f32> {
    let mut order = (0..rows).collect::<Vec<_>>();
    let mut rng = SplitMix64::new(seed);
    for last in (1..rows).rev() {
        order.swap(last, rng.index(last + 1));
    }
    if order.iter().enumerate().all(|(row, &source)| row == source) {
        order.rotate_left(1);
    }
    let mut shuffled = Vec::with_capacity(values.len());
    for source in order {
        shuffled.extend_from_slice(&values[source * columns..(source + 1) * columns]);
    }
    shuffled
}

fn seeds(seed: u64, domain: u64) -> u64 {
    let mut rng = SplitMix64::new(seed ^ domain.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    rng.next_u64()
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn signed(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    fn centered_oracle(values: &[f32], rows: usize) -> Vec<f32> {
        let left =
            (values.chunks_exact(2).map(|row| row[0] as f64).sum::<f64>() / rows as f64) as f32;
        let right =
            (values.chunks_exact(2).map(|row| row[1] as f64).sum::<f64>() / rows as f64) as f32;
        values
            .chunks_exact(2)
            .map(|row| (row[0] - left) * (row[1] - right))
            .collect()
    }

    #[test]
    fn materialization_is_row_major_and_matches_independent_mixed_oracle() {
        let values = [1.0, 3.0, 3.0, 1.0];
        let input = NativeMatrixRef::new(2, 2, &values).unwrap();
        let catalog = reference_catalog().unwrap();
        let bank = materialize_row_major(&input, &catalog).unwrap();
        assert_eq!(
            bank.values_row_major,
            [1.0, 2.0, 0.5, -1.0, 3.0, 2.0, 0.75, -1.0]
        );
        assert_eq!(bank.row(1), Some(&[3.0, 2.0, 0.75, -1.0][..]));
        let overflowed_slice = CandidateBank {
            rows: 1,
            candidate_count: usize::MAX,
            values_row_major: Vec::new(),
            candidate_major_bytes: 0,
            row_major_bytes: 0,
            conservative_peak_value_bytes: 0,
        };
        assert_eq!(overflowed_slice.row(1), None);
        assert_eq!(centered_oracle(&values, 2), [-1.0, -1.0]);
        assert_eq!(bank.conservative_peak_value_bytes, bank.row_major_bytes * 3);
    }

    #[test]
    fn parallel_gather_is_deterministic_and_pointwise_overflow_fails_closed() {
        let values = synthetic_inputs(97, 2, 41).unwrap();
        let input = NativeMatrixRef::new(97, 2, &values).unwrap();
        let catalog = reference_catalog().unwrap();
        let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let single = one.install(|| materialize_row_major(&input, &catalog).unwrap());
        let parallel = four.install(|| materialize_row_major(&input, &catalog).unwrap());
        assert_eq!(single, parallel);

        let overflow = [1.0, 0.0, f32::MAX, -f32::MAX];
        let overflow_input = NativeMatrixRef::new(2, 2, &overflow).unwrap();
        let absdiff = CandidateCatalog::new(
            2,
            vec![CandidateOp::AbsoluteDifference { left: 0, right: 1 }],
        )
        .unwrap();
        assert!(matches!(
            materialize_row_major(&overflow_input, &absdiff),
            Err(CandidateError::PointwiseNonFinite {
                candidate: 0,
                row: 1
            })
        ));
    }

    #[test]
    fn equal_scores_keep_the_first_catalog_index() {
        let tied = CandidateBank {
            rows: 3,
            candidate_count: 2,
            values_row_major: vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            candidate_major_bytes: 0,
            row_major_bytes: 0,
            conservative_peak_value_bytes: 0,
        };
        assert_eq!(
            select_train_view_consistency(&tied, &tied)
                .unwrap()
                .selected_index,
            0
        );
    }

    #[test]
    fn identity_control_reports_training_only_selection_and_holdout_scores() {
        let fixture = planted_fixture(73).unwrap();
        let train = NativeMatrixRef::new(
            fixture.train_rows,
            fixture.feature_count,
            &fixture.train_primary,
        )
        .unwrap();
        let aligned = NativeMatrixRef::new(
            fixture.train_rows,
            fixture.feature_count,
            &fixture.train_aligned,
        )
        .unwrap();
        let shuffled = NativeMatrixRef::new(
            fixture.train_rows,
            fixture.feature_count,
            &fixture.train_shuffled,
        )
        .unwrap();
        let train_bank = materialize_row_major(&train, &fixture.catalog).unwrap();
        let aligned_bank = materialize_row_major(&aligned, &fixture.catalog).unwrap();
        let shuffled_bank = materialize_row_major(&shuffled, &fixture.catalog).unwrap();
        let selection = select_train_view_consistency(&train_bank, &aligned_bank).unwrap();
        let shuffled_scores = paired_view_scores(&train_bank, &shuffled_bank).unwrap();
        assert!(selection
            .per_candidate_scores
            .iter()
            .all(|score| score.is_finite()));
        // Consistency may favor softsign over identity under finite noise.
        // This control checks evidence-based selection, not target-optimality.
        assert!(matches!(selection.selected_index, 0 | 2));
        assert!(selection.per_candidate_scores[selection.selected_index] > 0.9);
        assert!(
            selection.per_candidate_scores[selection.selected_index]
                > shuffled_scores[selection.selected_index] + 0.5
        );

        let holdout = NativeMatrixRef::new(
            fixture.holdout_rows,
            fixture.feature_count,
            &fixture.holdout_primary,
        )
        .unwrap();
        let holdout_bank = materialize_row_major(&holdout, &fixture.catalog).unwrap();
        let holdout_scores = holdout_target_scores(&holdout_bank, &fixture.holdout_target).unwrap();
        assert!(holdout_scores.iter().all(|score| score.is_finite()));
        assert!(holdout_scores[selection.selected_index] > 0.9);
    }

    #[test]
    fn invariance_fixture_selects_absdiff_before_holdout_target_is_scored() {
        let fixture = planted_invariance_fixture(73).unwrap();
        let bank = |values: &[f32]| {
            let input =
                NativeMatrixRef::new(fixture.train_rows, fixture.feature_count, values).unwrap();
            materialize_row_major(&input, &fixture.catalog).unwrap()
        };
        let train_bank = bank(&fixture.train_primary);
        let aligned_bank = bank(&fixture.train_aligned);
        let shuffled_bank = bank(&fixture.train_shuffled);
        let selection = select_train_view_consistency(&train_bank, &aligned_bank).unwrap();
        let shuffled_scores = paired_view_scores(&train_bank, &shuffled_bank).unwrap();
        assert_eq!(selection.selected_index, 1);
        assert!(selection
            .per_candidate_scores
            .iter()
            .all(|score| score.is_finite()));
        assert!(shuffled_scores[1] < selection.per_candidate_scores[1] - 0.5);

        let holdout_input = NativeMatrixRef::new(
            fixture.holdout_rows,
            fixture.feature_count,
            &fixture.holdout_primary,
        )
        .unwrap();
        let holdout_bank = materialize_row_major(&holdout_input, &fixture.catalog).unwrap();
        let scores = holdout_target_scores(&holdout_bank, &fixture.holdout_target).unwrap();
        assert!(scores.iter().all(|score| score.is_finite()));
        assert!(scores[1] > 0.9);
    }
}
