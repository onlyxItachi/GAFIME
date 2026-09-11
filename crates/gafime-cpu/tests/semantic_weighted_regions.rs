//! Focused Core regressions for canonical finite weighted region sums.
//!
//! These deliberately exercise the program form directly instead of a native
//! query: Core is the three-profile arithmetic reference, while local CUDA is
//! only an optional fp32 physical lowering.

use gafime_cpu::semantic::CoreEvidenceExecutor;
use gafime_orchestrator::semantic::{
    CandidateRegistry, EvaluationRole, FeatureFrame, FeatureId, FeatureOp, NativeEvidenceExecutor,
    NumericColumn, PredicateComparator, ProgramLimits, SemanticError,
};
use gafime_types::PrecisionProfile;

const BUDGET: usize = 1 << 20;

fn frame(profile: PrecisionProfile, values: Vec<f64>) -> FeatureFrame {
    let rows = values.len();
    let column = match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => NumericColumn::from(
            values
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ),
        PrecisionProfile::Fp64 => NumericColumn::from(values),
    };
    FeatureFrame::with_profile(
        profile,
        vec!["x".into()],
        "weighted-region-rows".into(),
        (0..rows as u64).collect(),
        EvaluationRole::Discovery,
        "weighted-region-core-test".into(),
        vec![column],
    )
    .unwrap()
}

fn two_regions(
    registry: &mut CandidateRegistry,
    profile: PrecisionProfile,
) -> (FeatureId, FeatureId) {
    let source = registry.source(0).unwrap();
    let (lower, upper) = match profile {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => (
            registry
                .hard_predicate(source, PredicateComparator::GreaterThan, 0.0)
                .unwrap(),
            registry
                .hard_predicate(source, PredicateComparator::GreaterThan, 1.0)
                .unwrap(),
        ),
        PrecisionProfile::Fp64 => (
            registry
                .hard_predicate_f64(source, PredicateComparator::GreaterThan, 0.0)
                .unwrap(),
            registry
                .hard_predicate_f64(source, PredicateComparator::GreaterThan, 1.0)
                .unwrap(),
        ),
    };
    (
        registry.decision_region(vec![lower]).unwrap(),
        registry.decision_region(vec![upper]).unwrap(),
    )
}

#[test]
fn weighted_regions_keep_region_weight_pairs_canonical_and_distinct_from_counts() {
    let mut registry = CandidateRegistry::new(
        vec!["x".into()],
        PrecisionProfile::Fp32,
        ProgramLimits::default(),
    )
    .unwrap();
    let source = registry.source(0).unwrap();
    let (first, second) = two_regions(&mut registry, PrecisionProfile::Fp32);

    let weighted = registry
        .region_weighted_sum(vec![second, first], vec![2.0, -0.0])
        .unwrap();
    assert_eq!(
        registry
            .region_weighted_sum(vec![first, second], vec![-0.0, 2.0])
            .unwrap(),
        weighted,
        "canonical region ordering must retain its paired weight"
    );
    let FeatureOp::RegionWeightedSum {
        regions,
        weight_bits,
    } = registry.program(weighted).unwrap().op()
    else {
        panic!("expected canonical weighted region sum");
    };
    assert_eq!(regions, &[first, second]);
    assert_eq!(
        weight_bits.as_f32_bits().unwrap(),
        &[(-0.0f32).to_bits(), 2.0f32.to_bits()]
    );
    assert_ne!(
        registry
            .region_weighted_sum(vec![first, second], vec![2.0, -0.0])
            .unwrap(),
        weighted,
        "sorting regions must not detach or reorder the supplied weight pairs"
    );
    let all_ones = registry
        .region_weighted_sum(vec![first, second], vec![1.0, 1.0])
        .unwrap();
    assert_ne!(
        registry.region_count(vec![first, second]).unwrap(),
        all_ones,
        "all-one weighted arithmetic remains a distinct program form from RegionCount"
    );

    for (regions, weights) in [
        (vec![first], vec![1.0]),
        (vec![first, first], vec![1.0, 2.0]),
        (vec![first, second], vec![1.0]),
        (vec![first, second], vec![f32::NAN, 1.0]),
        (vec![first, second], vec![f32::INFINITY, 1.0]),
        (vec![source, first], vec![1.0, 2.0]),
    ] {
        assert!(registry.region_weighted_sum(regions, weights).is_err());
    }
}

#[test]
fn core_weighted_regions_use_profile_native_ordered_addition_and_fail_closed_on_overflow() {
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(profile, vec![-1.0, 0.0, 0.5, 1.5, 4.0]);
        let mut registry =
            CandidateRegistry::new(input.schema().to_vec(), profile, ProgramLimits::default())
                .unwrap();
        let (first, second) = two_regions(&mut registry, profile);
        let weighted = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => registry
                .region_weighted_sum(vec![second, first], vec![2.5, -0.75])
                .unwrap(),
            PrecisionProfile::Fp64 => {
                let adjacent_one = f64::from_bits(1.0f64.to_bits() + 1);
                registry
                    .region_weighted_sum_f64(vec![second, first], vec![-0.5, adjacent_one])
                    .unwrap()
            }
        };
        let mut core = CoreEvidenceExecutor::default();
        let output = core
            .materialize(&registry, &input, &[weighted], None, BUDGET)
            .unwrap();
        match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => assert_eq!(
                output.get_typed(weighted).unwrap().as_f32().unwrap(),
                &[0.0, 0.0, -0.75, 1.75, 1.75],
            ),
            PrecisionProfile::Fp64 => {
                let adjacent_one = f64::from_bits(1.0f64.to_bits() + 1);
                assert_eq!(
                    output.get_typed(weighted).unwrap().as_f64().unwrap(),
                    &[
                        0.0,
                        0.0,
                        adjacent_one,
                        adjacent_one - 0.5,
                        adjacent_one - 0.5
                    ],
                    "fp64 weights must not be rounded through the fp32 lane",
                );
            }
        }
    }

    // Discovery frames require at least two rows. Both rows satisfy both
    // nested regions, so each follows the same overflowing ordered sum.
    // The post-node finite gate is shared by all three declared arithmetic
    // lanes, but exercise each lane so an accidental f32 narrowing of fp64
    // cannot hide behind the common diagnostic.
    for profile in [
        PrecisionProfile::Fp32,
        PrecisionProfile::Mixed,
        PrecisionProfile::Fp64,
    ] {
        let input = frame(profile, vec![2.0, 2.0]);
        let mut registry =
            CandidateRegistry::new(input.schema().to_vec(), profile, ProgramLimits::default())
                .unwrap();
        let (first, second) = two_regions(&mut registry, profile);
        let overflow = match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => registry
                .region_weighted_sum(vec![first, second], vec![f32::MAX, f32::MAX])
                .unwrap(),
            PrecisionProfile::Fp64 => registry
                .region_weighted_sum_f64(vec![first, second], vec![f64::MAX, f64::MAX])
                .unwrap(),
        };
        let error = match CoreEvidenceExecutor::default().materialize(
            &registry,
            &input,
            &[overflow],
            None,
            BUDGET,
        ) {
            Ok(_) => panic!("overflowing weighted sum must not materialize"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            SemanticError::Invalid(
                "candidate arithmetic produced nonfinite or profile-incompatible values"
            ),
            "finite profile-native weights whose declared accumulation overflows must fail closed"
        );
    }
}
