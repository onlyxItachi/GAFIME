//! Independent numerical regressions for the closed semantic association
//! vocabulary. These assertions intentionally do not reuse the semantic
//! implementation's rank, histogram, or corrected-MI helpers.

use std::sync::Arc;

use gafime_cpu::{
    kernels::precision::{
        mutual_info_fixed_f32, mutual_info_fixed_f64, mutual_info_fixed_mixed, spearman_f32,
        spearman_f64, spearman_mixed,
    },
    semantic::CoreEvidenceExecutor,
};
use gafime_orchestrator::semantic::{
    AssociationContext, AssociationStatistic, CandidateRegistry, EvaluationRole, EvidenceChannel,
    EvidenceDefinition, EvidenceRecord, EvidenceTable, EvidenceValue, FeatureFrame, FeatureId,
    LabelSet, NumericColumn, ProgramLimits, SemanticSession, UnavailableReason,
};
use gafime_types::{PrecisionProfile, GAFIME_BACKEND_CPU};

const EXECUTION_BUDGET: usize = 8 * 1024 * 1024;

fn frame(
    profile: PrecisionProfile,
    schema: &[&str],
    values: Vec<Vec<f64>>,
    provenance: &str,
) -> Arc<FeatureFrame> {
    let rows = values.first().expect("fixture has a column").len();
    let columns = match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => values
            .into_iter()
            .map(|column| {
                NumericColumn::from(
                    column
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
        PrecisionProfile::Fp64 => values.into_iter().map(NumericColumn::from).collect(),
    };
    Arc::new(
        FeatureFrame::with_profile(
            profile,
            schema.iter().map(|name| (*name).to_owned()).collect(),
            "semantic-association-rows".into(),
            (0..u64::try_from(rows).expect("fixture row count fits u64")).collect(),
            EvaluationRole::Discovery,
            provenance.into(),
            columns,
        )
        .unwrap(),
    )
}

fn session(input: &FeatureFrame) -> SemanticSession {
    let registry = CandidateRegistry::new(
        input.schema().to_vec(),
        input.profile(),
        ProgramLimits::default(),
    )
    .unwrap();
    SemanticSession::new(registry, GAFIME_BACKEND_CPU, EXECUTION_BUDGET).unwrap()
}

fn association(statistic: AssociationStatistic, context: AssociationContext) -> EvidenceDefinition {
    EvidenceDefinition::Association { statistic, context }
}

fn fixed_nmi(reference: FeatureId, bins: u32) -> EvidenceDefinition {
    association(
        AssociationStatistic::FixedCorrectedNmi { bins },
        AssociationContext::Reference { reference },
    )
}

fn measured(
    table: &EvidenceTable,
    candidate: FeatureId,
    channel: &EvidenceChannel,
) -> (f64, usize) {
    match table.value(candidate, channel.id()).unwrap() {
        EvidenceValue::Measured { value, support } => (value, support),
        value => panic!("expected measured evidence, got {value:?}"),
    }
}

fn unavailable(value: EvidenceValue, expected_reason: UnavailableReason, expected_support: usize) {
    assert_eq!(
        value,
        EvidenceValue::Unavailable {
            reason: expected_reason,
            support: expected_support,
        }
    );
}

fn partial_labels(frame: &FeatureFrame, values: &[f64]) -> Arc<LabelSet> {
    match frame.profile() {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => Arc::new(
            LabelSet::new(
                frame,
                values
                    .iter()
                    .enumerate()
                    .step_by(2)
                    .map(|(row, value)| (row, *value as f32))
                    .collect(),
                "partial association labels".into(),
            )
            .unwrap(),
        ),
        PrecisionProfile::Fp64 => Arc::new(
            LabelSet::new_f64(
                frame,
                values.iter().copied().enumerate().step_by(2).collect(),
                "partial association labels".into(),
            )
            .unwrap(),
        ),
    }
}

#[derive(Debug)]
struct NmiOracle {
    value: f64,
    hist_x: Vec<u32>,
    hist_y: Vec<u32>,
    joint: Vec<u32>,
}

fn fixed_bin_f32(scaled: f32, bins: usize) -> usize {
    let max = bins - 1;
    if scaled.is_nan() || scaled <= 0.0 {
        0
    } else if !scaled.is_finite() || scaled >= max as f32 {
        max
    } else {
        scaled as usize
    }
}

fn fixed_bin_f64(scaled: f64, bins: usize) -> usize {
    let max = bins - 1;
    if scaled.is_nan() || scaled <= 0.0 {
        0
    } else if !scaled.is_finite() || scaled >= max as f64 {
        max
    } else {
        scaled as usize
    }
}

fn histogram_f32(x: &[f32], y: &[f32], bins: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let min_x = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_y = y.iter().copied().fold(f32::INFINITY, f32::min);
    let max_y = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_x = bins as f32 / (max_x - min_x);
    let inv_y = bins as f32 / (max_y - min_y);
    let mut hist_x = vec![0u32; bins];
    let mut hist_y = vec![0u32; bins];
    let mut joint = vec![0u32; bins * bins];
    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_bin = fixed_bin_f32((x_value - min_x) * inv_x, bins);
        let y_bin = fixed_bin_f32((y_value - min_y) * inv_y, bins);
        hist_x[x_bin] += 1;
        hist_y[y_bin] += 1;
        joint[x_bin * bins + y_bin] += 1;
    }
    (hist_x, hist_y, joint)
}

fn histogram_f64(x: &[f64], y: &[f64], bins: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let min_x = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let inv_x = bins as f64 / (max_x - min_x);
    let inv_y = bins as f64 / (max_y - min_y);
    let mut hist_x = vec![0u32; bins];
    let mut hist_y = vec![0u32; bins];
    let mut joint = vec![0u32; bins * bins];
    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_bin = fixed_bin_f64((x_value - min_x) * inv_x, bins);
        let y_bin = fixed_bin_f64((y_value - min_y) * inv_y, bins);
        hist_x[x_bin] += 1;
        hist_y[y_bin] += 1;
        joint[x_bin * bins + y_bin] += 1;
    }
    (hist_x, hist_y, joint)
}

fn corrected_nmi_f32(hist_x: &[u32], hist_y: &[u32], joint: &[u32]) -> f32 {
    let bins = hist_x.len();
    let total = joint.iter().copied().sum::<u32>();
    let active_x = hist_x.iter().filter(|&&count| count != 0).count() as u32;
    let active_y = hist_y.iter().filter(|&&count| count != 0).count() as u32;
    assert!(total > 0 && active_x >= 2 && active_y >= 2);
    let total_f = total as f32;
    let mut mi = 0.0f32;
    for row in 0..bins {
        for col in 0..bins {
            let count = joint[row * bins + col];
            if count == 0 {
                continue;
            }
            let pxy = count as f32 / total_f;
            let px = hist_x[row] as f32 / total_f;
            let py = hist_y[col] as f32 / total_f;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    let correction = ((active_x - 1) as f32 * (active_y - 1) as f32) / (2.0 * total_f);
    let corrected = (mi - correction).max(0.0);
    corrected / (active_x.min(active_y) as f32).ln()
}

fn corrected_nmi_f64(hist_x: &[u32], hist_y: &[u32], joint: &[u32]) -> f64 {
    let bins = hist_x.len();
    let total = joint.iter().copied().sum::<u32>();
    let active_x = hist_x.iter().filter(|&&count| count != 0).count() as u32;
    let active_y = hist_y.iter().filter(|&&count| count != 0).count() as u32;
    assert!(total > 0 && active_x >= 2 && active_y >= 2);
    let total_f = total as f64;
    let mut mi = 0.0f64;
    for row in 0..bins {
        for col in 0..bins {
            let count = joint[row * bins + col];
            if count == 0 {
                continue;
            }
            let pxy = count as f64 / total_f;
            let px = hist_x[row] as f64 / total_f;
            let py = hist_y[col] as f64 / total_f;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    let correction = ((active_x - 1) as f64 * (active_y - 1) as f64) / (2.0 * total_f);
    let corrected = (mi - correction).max(0.0);
    corrected / (active_x.min(active_y) as f64).ln()
}

fn nmi_oracle(profile: PrecisionProfile, x: &[f64], y: &[f64], bins: usize) -> NmiOracle {
    match profile {
        PrecisionProfile::Fp32 => {
            let x = x.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let y = y.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let (hist_x, hist_y, joint) = histogram_f32(&x, &y, bins);
            NmiOracle {
                value: f64::from(corrected_nmi_f32(&hist_x, &hist_y, &joint)),
                hist_x,
                hist_y,
                joint,
            }
        }
        PrecisionProfile::Mixed => {
            let x = x.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let y = y.iter().map(|value| *value as f32).collect::<Vec<_>>();
            let (hist_x, hist_y, joint) = histogram_f32(&x, &y, bins);
            NmiOracle {
                value: corrected_nmi_f64(&hist_x, &hist_y, &joint),
                hist_x,
                hist_y,
                joint,
            }
        }
        PrecisionProfile::Fp64 => {
            let (hist_x, hist_y, joint) = histogram_f64(x, y, bins);
            NmiOracle {
                value: corrected_nmi_f64(&hist_x, &hist_y, &joint),
                hist_x,
                hist_y,
                joint,
            }
        }
    }
}

fn permutation_fixture(bins: usize) -> (Vec<f64>, Vec<f64>) {
    let rows = 8 * bins * bins;
    let x = (0..rows).map(|row| (row % bins) as f64).collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|value| ((3 * *value as usize) % bins) as f64)
        .collect::<Vec<_>>();
    (x, y)
}

fn nmi_evidence(
    profile: PrecisionProfile,
    candidate_values: Vec<f64>,
    reference_values: Vec<f64>,
    bins: u32,
) -> EvidenceValue {
    let input = frame(
        profile,
        &["candidate", "reference"],
        vec![candidate_values, reference_values],
        "fixed-NMI availability fixture",
    );
    let mut semantic = session(&input);
    let (candidate, reference) = {
        let round = semantic.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let channel = EvidenceChannel::new("fixed NMI".into(), fixed_nmi(reference, bins)).unwrap();
    let table = semantic
        .evaluate(
            &mut CoreEvidenceExecutor::default(),
            input,
            &[candidate],
            std::slice::from_ref(&channel),
        )
        .unwrap();
    table.value(candidate, channel.id()).unwrap()
}

#[test]
fn fixed_nmi_matches_an_independent_fixed_histogram_oracle_at_four_and_eight_bins() {
    for bins in [4usize, 8] {
        let (candidate_values, reference_values) = permutation_fixture(bins);
        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let oracle = nmi_oracle(profile, &candidate_values, &reference_values, bins);
            let per_active_bin = u32::try_from(candidate_values.len() / bins).unwrap();
            assert_eq!(oracle.hist_x, vec![per_active_bin; bins]);
            assert_eq!(oracle.hist_y, vec![per_active_bin; bins]);
            for x_bin in 0..bins {
                for y_bin in 0..bins {
                    let expected = if y_bin == (3 * x_bin) % bins {
                        per_active_bin
                    } else {
                        0
                    };
                    assert_eq!(oracle.joint[x_bin * bins + y_bin], expected);
                }
            }

            let input = frame(
                profile,
                &["candidate", "reference"],
                vec![candidate_values.clone(), reference_values.clone()],
                "fixed-NMI oracle fixture",
            );
            let mut semantic = session(&input);
            let (candidate, reference) = {
                let round = semantic.begin_round(&[]).unwrap();
                (round.source(0).unwrap(), round.source(1).unwrap())
            };
            let channel = EvidenceChannel::new(
                format!("fixed NMI {bins}"),
                fixed_nmi(reference, u32::try_from(bins).unwrap()),
            )
            .unwrap();
            let table = semantic
                .evaluate(
                    &mut CoreEvidenceExecutor::default(),
                    input,
                    &[candidate],
                    std::slice::from_ref(&channel),
                )
                .unwrap();
            let (actual, support) = measured(&table, candidate, &channel);
            assert_eq!(support, candidate_values.len());
            assert_eq!(
                actual.to_bits(),
                oracle.value.to_bits(),
                "{profile:?} bins={bins} must retain the fixed-bin histogram and profile lane"
            );
        }
    }
}

#[test]
fn tied_rank_contexts_preserve_signed_pairing_and_partial_label_scope_in_every_profile() {
    let rows = 512usize;
    let candidate = (0..rows).map(|row| (row % 16) as f64).collect::<Vec<_>>();
    let reference = candidate
        .iter()
        .map(|value| value.powi(3))
        .collect::<Vec<_>>();
    let paired_candidate = candidate
        .iter()
        .map(|value| 15.0 - value)
        .collect::<Vec<_>>();
    let label_values = paired_candidate.clone();

    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(
            profile,
            &["candidate", "reference"],
            vec![candidate.clone(), reference.clone()],
            "tied rank discovery fixture",
        );
        let paired = frame(
            profile,
            &["candidate", "reference"],
            vec![paired_candidate.clone(), reference.clone()],
            "tied rank aligned fixture",
        );
        let labels = partial_labels(&input, &label_values);
        let mut semantic = session(&input);
        let (candidate_id, reference_id) = {
            let round = semantic.begin_round(&[]).unwrap();
            (round.source(0).unwrap(), round.source(1).unwrap())
        };
        let reference_channel = EvidenceChannel::new(
            "absolute tied-rank reference".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::Reference {
                    reference: reference_id,
                },
            ),
        )
        .unwrap();
        let paired_channel = EvidenceChannel::new(
            "signed tied-rank paired view".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::PairedView {
                    view: Arc::clone(&paired),
                },
            ),
        )
        .unwrap();
        let labels_channel = EvidenceChannel::new(
            "absolute tied-rank labels".into(),
            association(
                AssociationStatistic::Spearman,
                AssociationContext::Labels {
                    labels: Some(labels),
                },
            ),
        )
        .unwrap();
        let table = semantic
            .evaluate(
                &mut CoreEvidenceExecutor::default(),
                input,
                &[candidate_id],
                &[
                    reference_channel.clone(),
                    paired_channel.clone(),
                    labels_channel.clone(),
                ],
            )
            .unwrap();
        let (reference_value, reference_support) =
            measured(&table, candidate_id, &reference_channel);
        let (paired_value, paired_support) = measured(&table, candidate_id, &paired_channel);
        let (labels_value, labels_support) = measured(&table, candidate_id, &labels_channel);
        let tolerance = match profile {
            PrecisionProfile::Fp32 => 1.0e-6,
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => 1.0e-12,
        };
        assert!((reference_value - 1.0).abs() <= tolerance);
        assert!((paired_value + 1.0).abs() <= tolerance);
        assert!((labels_value - 1.0).abs() <= tolerance);
        assert_eq!(reference_support, rows);
        assert_eq!(paired_support, rows);
        assert_eq!(labels_support, rows / 2);
    }
}

#[test]
fn fixed_nmi_keeps_invalid_and_zero_dependence_outcomes_distinct() {
    let bins = 4u32;
    let mut independent_x = Vec::new();
    let mut independent_y = Vec::new();
    for _ in 0..8 {
        for x in 0..bins {
            for y in 0..bins {
                independent_x.push(f64::from(x));
                independent_y.push(f64::from(y));
            }
        }
    }

    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let independent = nmi_evidence(profile, independent_x.clone(), independent_y.clone(), bins);
        assert_eq!(
            independent,
            EvidenceValue::Measured {
                value: 0.0,
                support: independent_x.len(),
            },
            "{profile:?} independence is measured zero, not unavailable evidence"
        );

        unavailable(
            nmi_evidence(
                profile,
                independent_x[..127].to_vec(),
                independent_y[..127].to_vec(),
                bins,
            ),
            UnavailableReason::InsufficientSupport,
            127,
        );
        unavailable(
            nmi_evidence(
                profile,
                vec![1.0; independent_x.len()],
                independent_y.clone(),
                bins,
            ),
            UnavailableReason::ConstantOperand,
            independent_x.len(),
        );

        let extreme = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => f64::from(f32::MAX),
            PrecisionProfile::Fp64 => f64::MAX,
        };
        let overflow_x = (0..independent_x.len())
            .map(|row| if row % 2 == 0 { -extreme } else { extreme })
            .collect::<Vec<_>>();
        let overflow_y = overflow_x.iter().map(|value| -*value).collect::<Vec<_>>();
        unavailable(
            nmi_evidence(profile, overflow_x, overflow_y, bins),
            UnavailableReason::NonFiniteReduction,
            independent_x.len(),
        );
    }

    for profile in [PrecisionProfile::Fp32, PrecisionProfile::Mixed] {
        let storage_collapsed = (0..independent_x.len())
            .map(|row| 1.0 + f64::EPSILON * row as f64)
            .collect::<Vec<_>>();
        unavailable(
            nmi_evidence(profile, storage_collapsed, independent_y.clone(), bins),
            UnavailableReason::ConstantOperand,
            independent_x.len(),
        );
    }
}

fn worker_records(workers: usize) -> Vec<EvidenceValue> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap()
        .install(|| {
            let (candidate_values, reference_values) = permutation_fixture(8);
            let input = frame(
                PrecisionProfile::Mixed,
                &["candidate", "reference"],
                vec![candidate_values, reference_values],
                "association worker determinism fixture",
            );
            let mut semantic = session(&input);
            let (candidate, derived, reference) = {
                let mut round = semantic.begin_round(&[]).unwrap();
                let candidate = round.source(0).unwrap();
                let reference = round.source(1).unwrap();
                let derived = round.softsign(candidate).unwrap();
                (candidate, derived, reference)
            };
            let rank = EvidenceChannel::new(
                "rank worker channel".into(),
                association(
                    AssociationStatistic::Spearman,
                    AssociationContext::Reference { reference },
                ),
            )
            .unwrap();
            let nmi =
                EvidenceChannel::new("NMI worker channel".into(), fixed_nmi(reference, 8)).unwrap();
            let table = semantic
                .evaluate(
                    &mut CoreEvidenceExecutor::default(),
                    input,
                    &[candidate, derived],
                    &[rank, nmi],
                )
                .unwrap();
            table.records().iter().map(EvidenceRecord::value).collect()
        })
}

#[test]
fn association_identity_duplicate_work_and_parallel_outputs_are_deterministic() {
    let (candidate_values, reference_values) = permutation_fixture(4);
    let input = frame(
        PrecisionProfile::Mixed,
        &["candidate", "reference"],
        vec![candidate_values, reference_values],
        "association identity fixture",
    );
    let mut semantic = session(&input);
    let (candidate, reference) = {
        let round = semantic.begin_round(&[]).unwrap();
        (round.source(0).unwrap(), round.source(1).unwrap())
    };
    let first = EvidenceChannel::new("first fixed NMI".into(), fixed_nmi(reference, 4)).unwrap();
    let duplicate =
        EvidenceChannel::new("duplicate fixed NMI".into(), fixed_nmi(reference, 4)).unwrap();
    assert!(first.rebind(fixed_nmi(reference, 8)).is_err());
    let mut core = CoreEvidenceExecutor::default();
    let calls_before = core.evidence_kernel_calls();
    let table = semantic
        .evaluate(
            &mut core,
            input,
            &[candidate],
            &[first.clone(), duplicate.clone()],
        )
        .unwrap();
    assert_eq!(core.evidence_kernel_calls() - calls_before, 1);
    assert_eq!(
        table.value(candidate, first.id()).unwrap(),
        table.value(candidate, duplicate.id()).unwrap()
    );
    assert_eq!(worker_records(1), worker_records(4));
}

#[test]
fn public_legacy_statistic_bits_remain_unchanged_for_ties_and_degenerate_inputs() {
    let x_f32 = [1.0f32, 1.0, 2.0, 5.0];
    let y_f32 = [4.0f32, 4.0, 3.0, 2.0];
    let x_f64 = x_f32.map(f64::from);
    let y_f64 = y_f32.map(f64::from);
    assert_eq!(spearman_f32(&x_f32, &y_f32).to_bits(), (-1.0f32).to_bits());
    assert_eq!(
        spearman_mixed(&x_f32, &y_f32).to_bits(),
        (-1.0f64).to_bits()
    );
    assert_eq!(spearman_f64(&x_f64, &y_f64).to_bits(), (-1.0f64).to_bits());

    let constant_f32 = [1.0f32, 1.0, 1.0, 1.0];
    let varying_f32 = [0.0f32, 1.0, 0.0, 1.0];
    let constant_f64 = constant_f32.map(f64::from);
    let varying_f64 = varying_f32.map(f64::from);
    assert_eq!(
        mutual_info_fixed_f32(&constant_f32, &varying_f32, 4).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        mutual_info_fixed_mixed(&constant_f32, &varying_f32, 4).to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(
        mutual_info_fixed_f64(&constant_f64, &varying_f64, 4).to_bits(),
        0.0f64.to_bits()
    );
}
