//! Direct native-boundary parity; public FeatureFrame still rejects nonfinite
//! inputs. A configured payload is required to execute, not silently skipped.
use gafime_gpu_sys::{
    GpuBackend, SemanticFrozenRegionTerm, SemanticProgramNode, SemanticRegionRelation,
    CUDA_LIBRARY_ENV, METAL_LIBRARY_ENV, ROCM_LIBRARY_ENV,
};
use gafime_types::PrecisionProfile;

fn assert_predicate_definedness(backend: &GpuBackend, profiles: &[PrecisionProfile]) {
    for &profile in profiles {
        for reverse in [false, true] {
            let bank = backend.allocate_semantic_bank(profile, 6, 2, 3).unwrap();
            let values = [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::NAN,
                0.0,
                -0.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
            ];
            if profile == PrecisionProfile::Fp64 {
                bank.upload_f64(&values).unwrap();
            } else {
                bank.upload_f32(&values.map(|value| value as f32)).unwrap();
            }
            let mut terms = vec![
                SemanticFrozenRegionTerm {
                    input_slot: 0,
                    relation: SemanticRegionRelation::GreaterThan,
                    threshold_bits: 0,
                },
                SemanticFrozenRegionTerm {
                    input_slot: 1,
                    relation: SemanticRegionRelation::GreaterThan,
                    threshold_bits: 0,
                },
            ];
            if reverse {
                terms.reverse();
            }
            bank.materialize(&[SemanticProgramNode::FrozenRegionConjunction {
                output_slot: 2,
                terms,
            }])
            .expect("false dominates NaN; infinities retain ordinary ordered comparison");
            let actual = if profile == PrecisionProfile::Fp64 {
                bank.download_f64(&[2]).unwrap()
            } else {
                bank.download_f32(&[2])
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect()
            };
            assert_eq!(actual, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let bank = backend.allocate_semantic_bank(profile, 1, 1, 2).unwrap();
        if profile == PrecisionProfile::Fp64 {
            bank.upload_f64(&[f64::NAN]).unwrap();
        } else {
            bank.upload_f32(&[f32::NAN]).unwrap();
        }
        assert!(
            bank.materialize(&[SemanticProgramNode::FrozenRegionConjunction {
                output_slot: 1,
                terms: vec![SemanticFrozenRegionTerm {
                    input_slot: 0,
                    relation: SemanticRegionRelation::LessEqual,
                    threshold_bits: 0,
                }],
            }])
            .is_err(),
            "an unresolved final membership must still fail closed"
        );
        assert!(
            if profile == PrecisionProfile::Fp64 {
                bank.download_f64(&[1]).is_err()
            } else {
                bank.download_f32(&[1]).is_err()
            },
            "failed materialization must not initialize an output slot"
        );
    }
}

#[test]
fn cuda_semantic_predicate_definedness_when_configured() {
    if std::env::var_os(CUDA_LIBRARY_ENV).is_none() {
        return;
    }
    assert_predicate_definedness(
        &GpuBackend::cuda_from_env(0).unwrap(),
        &[
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ],
    );
}

#[test]
fn rocm_semantic_predicate_definedness_when_configured() {
    if std::env::var_os(ROCM_LIBRARY_ENV).is_none() {
        return;
    }
    assert_predicate_definedness(
        &GpuBackend::rocm_from_env(0).unwrap(),
        &[
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ],
    );
}

#[test]
fn metal_semantic_predicate_definedness_when_configured() {
    if std::env::var_os(METAL_LIBRARY_ENV).is_none() {
        return;
    }
    assert_predicate_definedness(
        &GpuBackend::metal_from_env(0).unwrap(),
        &[PrecisionProfile::Fp32],
    );
}
