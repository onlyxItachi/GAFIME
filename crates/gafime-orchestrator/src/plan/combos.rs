use gafime_types::{
    BackendKind, GafimeArityChunk, GafimePermutationSchedule, GafimeRankSpec, GAFIME_BACKEND_METAL,
    GAFIME_FAMILY_CONTINUOUS,
};
use std::sync::Arc;

use crate::backend::{OrchestratorError, OrchestratorResult};

use super::{legacy_rng::PythonRandom, shapes, CompiledPlan, DEFAULT_DESCRIPTOR_BATCH_WORDS};

pub const MI_TEMPLATE_BIN_LEVELS: &[u32] = &[2, 4, 8, 12, 16, 24, 32, 48, 64, 96];
pub const MI_SAMPLES_PER_JOINT_BIN: u64 = 8;
pub const DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct CombinationDescriptorSource {
    unary_features: Arc<[u32]>,
    higher_features: Arc<[u32]>,
}

impl CombinationDescriptorSource {
    pub(crate) fn new(unary_features: &[u32], higher_features: &[u32]) -> Self {
        let unary_features: Arc<[u32]> = unary_features.into();
        let higher_features = if unary_features.as_ref() == higher_features {
            Arc::clone(&unary_features)
        } else {
            Arc::from(higher_features)
        };
        Self {
            unary_features,
            higher_features,
        }
    }

    pub(super) fn features_for_arity(&self, arity: u32) -> &[u32] {
        if arity == 1 {
            &self.unary_features
        } else {
            &self.higher_features
        }
    }

    pub(super) fn retained_word_count(&self) -> usize {
        if Arc::ptr_eq(&self.unary_features, &self.higher_features) {
            self.unary_features.len()
        } else {
            self.unary_features
                .len()
                .saturating_add(self.higher_features.len())
        }
    }

    pub(super) fn validate(&self, n_features: u32) -> OrchestratorResult<()> {
        validate_feature_order(&self.unary_features, n_features)?;
        if !Arc::ptr_eq(&self.unary_features, &self.higher_features) {
            validate_feature_order(&self.higher_features, n_features)?;
        }
        Ok(())
    }

    pub(super) fn materialize(&self, chunks: &[GafimeArityChunk]) -> Vec<u32> {
        let descriptor_words = chunks.iter().fold(0u64, |total, chunk| {
            total.saturating_add(chunk.combo_count.saturating_mul(u64::from(chunk.arity)))
        });
        let capacity = usize::try_from(descriptor_words)
            .expect("materialized descriptor buffer exceeds the host address space");
        let mut descriptors = Vec::with_capacity(capacity);
        for chunk in chunks {
            generate_combinations_from_features_limited(
                self.features_for_arity(chunk.arity),
                chunk.arity as usize,
                chunk.combo_count,
                &mut descriptors,
            );
        }
        debug_assert_eq!(descriptors.len(), capacity);
        descriptors
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuousPlanRequest {
    pub backend_kind: BackendKind,
    pub n_samples: u64,
    pub n_features: u32,
    pub max_arity: u32,
    pub max_combinations_per_arity: u64,
    pub metric_ids: Vec<u32>,
    pub mi_bins: u32,
    pub rank: GafimeRankSpec,
}

pub fn legacy_unary_feature_order(
    n_features: u32,
    max_combinations_per_arity: u64,
    random_seed_words: &[u32],
) -> Vec<u32> {
    let mut features = (0..n_features).collect::<Vec<_>>();
    if u64::from(n_features) > max_combinations_per_arity {
        PythonRandom::from_seed_words(random_seed_words).shuffle(&mut features);
        features.truncate(max_combinations_per_arity.min(usize::MAX as u64) as usize);
    }
    features
}

pub fn legacy_higher_feature_order(
    n_features: u32,
    max_combinations_per_arity: u64,
    top_features_for_higher_arity: u32,
    random_seed_words: &[u32],
    unary_strengths: &[(u32, f32)],
) -> Vec<u32> {
    let mut random = PythonRandom::from_seed_words(random_seed_words);
    if u64::from(n_features) > max_combinations_per_arity {
        let mut consumed_unary_order = (0..n_features).collect::<Vec<_>>();
        random.shuffle(&mut consumed_unary_order);
    }

    let mut ranked = unary_strengths.to_vec();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_features_for_higher_arity as usize);
    let mut selected = ranked
        .into_iter()
        .map(|(feature, _)| feature)
        .collect::<Vec<_>>();
    random.shuffle(&mut selected);
    selected
}

/// Precision-aware form of [`legacy_higher_feature_order`] for mixed/fp64
/// visible ranking scores. Structural feature identities and randomized
/// scheduling remain unchanged; only the score lane is binary64.
pub fn legacy_higher_feature_order_f64(
    n_features: u32,
    max_combinations_per_arity: u64,
    top_features_for_higher_arity: u32,
    random_seed_words: &[u32],
    unary_strengths: &[(u32, f64)],
) -> Vec<u32> {
    let mut random = PythonRandom::from_seed_words(random_seed_words);
    if u64::from(n_features) > max_combinations_per_arity {
        let mut consumed_unary_order = (0..n_features).collect::<Vec<_>>();
        random.shuffle(&mut consumed_unary_order);
    }

    let mut ranked = unary_strengths.to_vec();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_features_for_higher_arity as usize);
    let mut selected = ranked
        .into_iter()
        .map(|(feature, _)| feature)
        .collect::<Vec<_>>();
    random.shuffle(&mut selected);
    selected
}

/// Resolve the configured adaptive MI ceiling to a supported template capacity.
pub fn sanitize_mi_bins_for_backend(backend_kind: BackendKind, bins: u32) -> u32 {
    let backend_ceiling = if backend_kind == GAFIME_BACKEND_METAL {
        48
    } else {
        96
    };
    let requested_ceiling = bins.clamp(2, backend_ceiling);
    MI_TEMPLATE_BIN_LEVELS
        .iter()
        .copied()
        .take_while(|&level| level <= requested_ceiling)
        .last()
        .unwrap_or(2)
}

/// Select the largest fixed histogram shape whose expected joint-cell density
/// satisfies the v0.4.1 finite-sample guard. `mi_bins` remains a maximum.
pub fn select_adaptive_mi_bins_for_backend(
    backend_kind: BackendKind,
    n_samples: u64,
    max_bins: u32,
) -> u32 {
    let template_ceiling = sanitize_mi_bins_for_backend(backend_kind, max_bins);
    let mut selected = 2;
    for &level in MI_TEMPLATE_BIN_LEVELS {
        if level > template_ceiling {
            break;
        }
        let required = MI_SAMPLES_PER_JOINT_BIN.saturating_mul(u64::from(level).pow(2));
        if n_samples < required {
            break;
        }
        selected = level;
    }
    selected
}

pub fn build_continuous_plan(request: ContinuousPlanRequest) -> OrchestratorResult<CompiledPlan> {
    let features = (0..request.n_features).collect::<Vec<_>>();
    build_continuous_plan_for_feature_orders(request, &features, &features, true)
}

pub fn build_continuous_plan_for_feature_orders(
    request: ContinuousPlanRequest,
    unary_features: &[u32],
    higher_features: &[u32],
    include_unary: bool,
) -> OrchestratorResult<CompiledPlan> {
    if request.n_samples == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan requires samples",
        ));
    }
    if request.n_features == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan requires features",
        ));
    }
    if request.max_arity == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan requires non-zero max arity",
        ));
    }
    if request.metric_ids.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan requires metrics",
        ));
    }
    validate_feature_order(unary_features, request.n_features)?;
    validate_feature_order(higher_features, request.n_features)?;

    let requested_max_arity = request.max_arity.min(request.n_features);
    let mut chunks = Vec::new();
    let mut total_rows = 0u64;
    let mut total_descriptor_words = 0u64;
    let mut shape_hints = Vec::new();

    for arity in 1..=requested_max_arity {
        if arity == 1 && !include_unary {
            continue;
        }
        let feature_order = if arity == 1 {
            unary_features
        } else {
            higher_features
        };
        let planned_count = binomial_saturating_u128(feature_order.len() as u64, arity as u64);
        let limit = saturating_u64_offset(planned_count).min(request.max_combinations_per_arity);
        if limit == 0 {
            continue;
        }
        let descriptor_offset = total_descriptor_words;
        let descriptor_words =
            limit
                .checked_mul(u64::from(arity))
                .ok_or(OrchestratorError::InvalidPlan(
                    "continuous descriptor count overflows",
                ))?;
        total_descriptor_words = total_descriptor_words.checked_add(descriptor_words).ok_or(
            OrchestratorError::InvalidPlan("continuous descriptor count overflows"),
        )?;
        let shape_hint_index = shape_hints.len() as u32;
        let mut shape_hint = shapes::default_shape_hint(request.backend_kind, arity);
        // The sample-size-selected MI template travels in vendor_hint.
        shape_hint.vendor_hint = select_adaptive_mi_bins_for_backend(
            request.backend_kind,
            request.n_samples,
            request.mi_bins,
        );
        shape_hints.push(shape_hint);
        chunks.push(GafimeArityChunk {
            arity,
            family: GAFIME_FAMILY_CONTINUOUS,
            metric_mask: 0,
            shape_hint_index,
            combo_row_offset: total_rows,
            combo_count: limit,
            local_chunk_id: chunks.len() as u32,
            flags: 0,
            descriptor_offset,
            descriptor_count: limit,
        });
        total_rows = total_rows
            .checked_add(limit)
            .ok_or(OrchestratorError::InvalidPlan(
                "continuous candidate count overflows",
            ))?;
    }

    if chunks.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan generated no chunks",
        ));
    }

    let max_arity = chunks.iter().map(|chunk| chunk.arity).max().unwrap_or(1);

    let descriptor_source = CombinationDescriptorSource::new(unary_features, higher_features);
    let bounded_rank = request.rank.top_k > 0;
    if !bounded_rank
        && unranked_candidate_storage_bytes(
            total_rows,
            total_descriptor_words,
            max_arity,
            request.metric_ids.len() as u32,
        ) > DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES
    {
        return Err(OrchestratorError::Unsupported(
            "unranked continuous candidate storage exceeds the host-memory budget",
        ));
    }
    if !bounded_rank || total_descriptor_words <= DEFAULT_DESCRIPTOR_BATCH_WORDS as u64 {
        let combo_indices = descriptor_source.materialize(&chunks);
        Ok(CompiledPlan::from_parts(
            request.backend_kind,
            request.n_samples,
            request.n_features,
            max_arity,
            combo_indices,
            request.metric_ids,
            chunks,
            shape_hints,
            request.rank,
            GafimePermutationSchedule::default(),
        ))
    } else {
        Ok(CompiledPlan::from_combination_parts(
            request.backend_kind,
            request.n_samples,
            request.n_features,
            max_arity,
            descriptor_source,
            request.metric_ids,
            chunks,
            shape_hints,
            request.rank,
            GafimePermutationSchedule::default(),
        ))
    }
}

pub fn unranked_candidate_storage_bytes(
    rows: u64,
    descriptor_words: u64,
    max_arity: u32,
    metric_count: u32,
) -> u64 {
    const U32_BYTES: u64 = 4;
    const U64_BYTES: u64 = 8;
    let row_bytes = u64::from(max_arity)
        .saturating_mul(U32_BYTES)
        .saturating_add(u64::from(metric_count).saturating_mul(U32_BYTES))
        .saturating_add(U32_BYTES) // rank
        .saturating_add(U32_BYTES) // family
        .saturating_add(U64_BYTES) // candidate id
        .saturating_add(U32_BYTES); // row flags
    descriptor_words
        .saturating_mul(U32_BYTES)
        .saturating_add(rows.saturating_mul(row_bytes))
}

fn validate_feature_order(features: &[u32], n_features: u32) -> OrchestratorResult<()> {
    if features.iter().any(|&feature| feature >= n_features) {
        return Err(OrchestratorError::InvalidPlan(
            "continuous feature order references an unknown feature",
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    if features.iter().any(|feature| !seen.insert(*feature)) {
        return Err(OrchestratorError::InvalidPlan(
            "continuous feature order contains a duplicate feature",
        ));
    }
    Ok(())
}

pub fn binomial_saturating_u128(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        let numerator = (n - k + i) as u128;
        result = result.saturating_mul(numerator) / i as u128;
    }
    result
}

pub fn saturating_u64_offset(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

fn generate_combinations_from_features_limited(
    features: &[u32],
    arity: usize,
    limit: u64,
    out: &mut Vec<u32>,
) -> u64 {
    if arity == 0 || arity > features.len() || limit == 0 {
        return 0;
    }
    let mut combo: Vec<usize> = (0..arity).collect();
    let mut generated = 0u64;
    loop {
        out.extend(combo.iter().map(|&index| features[index]));
        generated += 1;
        if generated >= limit {
            break;
        }

        let mut pivot = arity;
        while pivot > 0 {
            pivot -= 1;
            if combo[pivot] != pivot + features.len() - arity {
                break;
            }
            if pivot == 0 {
                return generated;
            }
        }
        combo[pivot] += 1;
        for idx in pivot + 1..arity {
            combo[idx] = combo[idx - 1] + 1;
        }
    }
    generated
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{
        GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_ROCM, GAFIME_METRIC_PEARSON,
    };

    #[test]
    fn counts_without_materializing() {
        assert_eq!(binomial_saturating_u128(5, 2), 10);
        assert_eq!(binomial_saturating_u128(1_000_000, 0), 1);
        assert_eq!(saturating_u64_offset(u128::MAX), u64::MAX);
    }

    #[test]
    fn hundred_million_pairs_keep_feature_bounded_descriptor_metadata() {
        let mut higher_features =
            legacy_higher_feature_order(6, 3, 3, &[7], &[(4, 0.5), (0, 0.1), (5, 0.6)]);
        higher_features.extend((0..20_000).filter(|feature| !matches!(feature, 0 | 4 | 5)));
        let plan = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 32,
                n_features: 20_000,
                max_arity: 2,
                max_combinations_per_arity: 100_000_000,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec {
                    top_k: 32,
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

        assert_eq!(plan.planned_row_count(), 100_000_000);
        assert_eq!(plan.logical_descriptor_words(), 200_000_000);
        assert!(plan.uses_generated_descriptors());
        assert_eq!(plan.materialized_descriptor_words(), 0);
        assert_eq!(plan.retained_descriptor_metadata_words(), 20_000);
        assert_eq!(plan.protocol().combo_indices.len, 0);
        plan.validate().unwrap();
        let result_plan = crate::reduce::CompactResultTablePlan::for_plan(&plan).unwrap();
        assert_eq!(result_plan.planned_rows(), 100_000_000);
        assert_eq!(result_plan.capacity(), 32);
        assert!(result_plan.is_rank_compacted());

        let mut batches = plan.descriptor_batches(6).unwrap();
        let first = batches.next().unwrap();
        let second = batches.next().unwrap();
        assert_eq!(first.logical_row_offset(), 0);
        assert_eq!(first.combo_indices(), &[4, 0, 4, 5, 4, 1]);
        assert_eq!(second.logical_row_offset(), 3);
        assert_eq!(second.combo_indices(), &[4, 2, 4, 3, 4, 6]);
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn unranked_plans_do_not_cliff_at_the_descriptor_batch_threshold() {
        let build = |n_features| {
            build_continuous_plan(ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 2,
                n_features,
                max_arity: 1,
                max_combinations_per_arity: u64::MAX,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec::default(),
            })
            .unwrap()
        };
        let below = build(DEFAULT_DESCRIPTOR_BATCH_WORDS as u32);
        let above = build(DEFAULT_DESCRIPTOR_BATCH_WORDS as u32 + 1);

        assert_eq!(
            below.logical_descriptor_words(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS as u64
        );
        assert_eq!(
            above.logical_descriptor_words(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS as u64 + 1
        );
        assert!(!below.uses_generated_descriptors());
        assert!(!above.uses_generated_descriptors());
        assert_eq!(
            below.materialized_descriptor_words(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS
        );
        assert_eq!(
            above.materialized_descriptor_words(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS + 1
        );
    }

    #[test]
    fn hundred_million_unranked_pairs_fail_before_descriptor_materialization() {
        let higher_features = (0..20_000).collect::<Vec<_>>();
        let error = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 2,
                n_features: 20_000,
                max_arity: 2,
                max_combinations_per_arity: 100_000_000,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec::default(),
            },
            &[],
            &higher_features,
            false,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OrchestratorError::Unsupported(
                "unranked continuous candidate storage exceeds the host-memory budget"
            )
        );
        assert_eq!(
            unranked_candidate_storage_bytes(100_000_000, 200_000_000, 2, 1),
            4_000_000_000
        );
    }

    #[test]
    fn continuous_plan_uses_flat_descriptor_offsets_for_mixed_arities() {
        let plan = build_continuous_plan(ContinuousPlanRequest {
            backend_kind: GAFIME_BACKEND_CPU,
            n_samples: 32,
            n_features: 4,
            max_arity: 2,
            max_combinations_per_arity: 10,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            mi_bins: 96,
            rank: GafimeRankSpec::default(),
        })
        .unwrap();

        assert_eq!(plan.chunks().len(), 2);
        assert_eq!(plan.chunks()[0].arity, 1);
        assert_eq!(plan.chunks()[0].descriptor_offset, 0);
        assert_eq!(plan.chunks()[0].combo_count, 4);
        assert_eq!(plan.chunks()[1].arity, 2);
        assert_eq!(plan.chunks()[1].descriptor_offset, 4);
        assert_eq!(plan.chunks()[1].combo_count, 6);
        plan.validate().unwrap();
    }

    #[test]
    fn legacy_feature_orders_match_v047_seeded_planning() {
        let unary = legacy_unary_feature_order(6, 3, &[7]);
        assert_eq!(unary, vec![4, 0, 5]);
        let higher = legacy_higher_feature_order(6, 3, 3, &[7], &[(4, 0.5), (0, 0.1), (5, 0.6)]);
        assert_eq!(higher, vec![4, 0, 5]);

        let tied =
            legacy_higher_feature_order(4, 10, 3, &[7], &[(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]);
        assert_eq!(tied, vec![2, 0, 1]);
    }

    #[test]
    fn f64_screening_order_preserves_scores_that_collapse_in_f32() {
        let lower = 1.0f64;
        let higher = f64::from_bits(lower.to_bits() + 1);
        assert_eq!(lower as f32, higher as f32);

        let selected = legacy_higher_feature_order_f64(2, 2, 1, &[7], &[(0, lower), (1, higher)]);
        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn screened_plan_preserves_seeded_descriptor_order() {
        let plan = build_continuous_plan_for_feature_orders(
            ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: 32,
                n_features: 6,
                max_arity: 3,
                max_combinations_per_arity: 3,
                metric_ids: vec![GAFIME_METRIC_PEARSON],
                mi_bins: 96,
                rank: GafimeRankSpec::default(),
            },
            &[4, 0, 5],
            &[4, 0, 5],
            true,
        )
        .unwrap();

        assert_eq!(plan.combo_indices(), &[4, 0, 5, 4, 0, 4, 5, 0, 5, 4, 0, 5]);
        assert_eq!(
            plan.chunks()
                .iter()
                .map(|chunk| (chunk.arity, chunk.combo_count))
                .collect::<Vec<_>>(),
            vec![(1, 3), (2, 3), (3, 1)]
        );
        plan.validate().unwrap();
    }

    #[test]
    fn continuous_plan_carries_sample_size_selected_mi_template() {
        let plan = build_continuous_plan(ContinuousPlanRequest {
            backend_kind: GAFIME_BACKEND_CUDA,
            n_samples: 2_048,
            n_features: 4,
            max_arity: 2,
            max_combinations_per_arity: 10,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            mi_bins: 96,
            rank: GafimeRankSpec::default(),
        })
        .unwrap();
        assert!(!plan.shape_hints.is_empty());
        assert!(plan.shape_hints.iter().all(|hint| hint.vendor_hint == 16));
    }

    #[test]
    fn adaptive_mi_template_scales_with_samples_and_respects_maximum() {
        let thresholds = [
            (0, 2),
            (31, 2),
            (32, 2),
            (128, 4),
            (512, 8),
            (1_152, 12),
            (2_048, 16),
            (4_608, 24),
            (8_192, 32),
            (18_432, 48),
            (32_768, 64),
            (73_728, 96),
        ];
        for (samples, expected) in thresholds {
            assert_eq!(
                select_adaptive_mi_bins_for_backend(GAFIME_BACKEND_CUDA, samples, 96),
                expected,
                "n_samples={samples}"
            );
            assert_eq!(
                select_adaptive_mi_bins_for_backend(GAFIME_BACKEND_ROCM, samples, 96),
                expected,
                "n_samples={samples}"
            );
        }
        assert_eq!(
            select_adaptive_mi_bins_for_backend(GAFIME_BACKEND_CUDA, 100_000, 20),
            16
        );
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_CUDA, 20), 16);
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_ROCM, 24), 24);
    }

    #[test]
    fn cpu_and_metal_use_their_supported_adaptive_ceiling() {
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, 24), 24);
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_CPU, 20), 16);
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_METAL, 48), 48);
        assert_eq!(sanitize_mi_bins_for_backend(GAFIME_BACKEND_METAL, 96), 48);
        assert_eq!(
            select_adaptive_mi_bins_for_backend(GAFIME_BACKEND_METAL, 100_000, 96),
            48
        );
    }
}
