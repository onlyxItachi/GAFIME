use gafime_types::{
    BackendKind, GafimeArityChunk, GafimePermutationSchedule, GafimeRankSpec, GAFIME_BACKEND_METAL,
    GAFIME_FAMILY_CONTINUOUS,
};

use crate::backend::{OrchestratorError, OrchestratorResult};

use super::{legacy_rng::PythonRandom, shapes, CompiledPlan};

pub const MI_TEMPLATE_BIN_LEVELS: &[u32] = &[2, 4, 8, 12, 16, 24, 32, 48, 64, 96];
pub const MI_SAMPLES_PER_JOINT_BIN: u64 = 8;

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
    let mut combo_indices = Vec::new();
    let mut chunks = Vec::new();
    let mut total_rows = 0u64;
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
        let limit = saturating_u64_offset(planned_count)
            .min(request.max_combinations_per_arity)
            .min(usize::MAX as u64);
        if limit == 0 {
            continue;
        }
        let descriptor_offset = combo_indices.len() as u64;
        let generated = generate_combinations_from_features_limited(
            feature_order,
            arity as usize,
            limit as usize,
            &mut combo_indices,
        ) as u64;
        if generated == 0 {
            continue;
        }
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
            combo_count: generated,
            local_chunk_id: chunks.len() as u32,
            flags: 0,
            descriptor_offset,
            descriptor_count: generated,
        });
        total_rows = total_rows.saturating_add(generated);
    }

    if chunks.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous plan generated no chunks",
        ));
    }

    let max_arity = chunks.iter().map(|chunk| chunk.arity).max().unwrap_or(1);

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
    limit: usize,
    out: &mut Vec<u32>,
) -> usize {
    if arity == 0 || arity > features.len() || limit == 0 {
        return 0;
    }
    let mut combo: Vec<usize> = (0..arity).collect();
    let mut generated = 0usize;
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
