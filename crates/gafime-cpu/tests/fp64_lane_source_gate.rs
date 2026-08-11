#[test]
fn binary64_simd_source_uses_only_f64_vector_arithmetic() {
    let source = include_str!("../src/simd/covariance_f64.rs");
    let implementation = source
        .split_once("#[cfg(test)]")
        .map_or(source, |(implementation, _)| implementation);

    for required in [
        "_mm512_loadu_pd",
        "_mm512_mul_pd",
        "_mm256_loadu_pd",
        "_mm256_mul_pd",
        "_mm_loadu_pd",
        "_mm_mul_pd",
        "vld1q_f64",
        "vmulq_f64",
    ] {
        assert!(
            implementation.contains(required),
            "binary64 SIMD ladder must retain {required}"
        );
    }

    for forbidden in [
        "_ps(",
        "cvtps",
        " as f32",
        "Vec<f32>",
        "&[f32]",
        "_fmadd",
        "vfmaq_f64",
        "vmlaq_f64",
        ".mul_add(",
    ] {
        assert!(
            !implementation.contains(forbidden),
            "binary64 SIMD source must not contain narrower operation {forbidden}"
        );
    }

    assert!(implementation.contains("covariance_common::"));
    assert!(implementation.contains("isa::finite_dispatch_isa"));
    assert!(
        !implementation.contains("struct EqualVectorParts"),
        "fp64 SIMD must reuse the shared non-floating vector partition scaffold"
    );
}

#[test]
fn public_fp64_pearson_is_only_a_binary64_simd_adapter() {
    let source = include_str!("../src/kernels/precision.rs");
    let (_, body) = source
        .split_once("pub fn pearson_f64")
        .expect("public fp64 Pearson function");
    let (body, _) = body
        .split_once("pub fn spearman_f32")
        .expect("public fp64 Pearson function end");

    assert!(body.contains("crate::simd::pearson_corr_f64(x, y)"));
    for forbidden in [" as f32", "Vec<f32>", "pearson_corr_f32", "pearson_sums("] {
        assert!(
            !body.contains(forbidden),
            "public fp64 Pearson adapter must not contain {forbidden}"
        );
    }
}

#[test]
fn fp64_r2_reuses_the_binary64_pearson_result() {
    let source = include_str!("../src/kernels/precision.rs");
    let (_, body) = source
        .split_once("fn score_f64")
        .expect("fp64 multi-metric scorer");
    let (body, _) = body
        .split_once("fn finalize_r2_f32")
        .expect("fp64 multi-metric scorer end");

    assert!(body.contains("get_or_insert_with(|| pearson_f64(signal, target))"));
    assert!(body.contains("crate::simd::finalize_r2_f64(corr)"));
    for forbidden in ["pearson_f32", "pearson_mixed", "finalize_r2_f32"] {
        assert!(
            !body.contains(forbidden),
            "fp64 R2 must not contain {forbidden}"
        );
    }
}

#[test]
fn binary64_finalization_and_tolerance_have_one_shared_contract() {
    let common = include_str!("../src/simd/covariance_common.rs");
    let precision = include_str!("../src/kernels/precision.rs");

    assert!(common.contains("pub(crate) const FP64_SIMD_REGROUPING_TOLERANCE: f64 = 2.0e-12"));
    assert!(common.contains("pub(crate) fn finalize_correlation_f64"));
    assert!(common.contains("pub(crate) fn finalize_r2_f64"));
    assert!(precision.contains("crate::simd::finalize_correlation_f64"));
    assert!(precision.contains("crate::simd::finalize_r2_f64"));
}

#[test]
fn mixed_and_fp64_spearman_finalize_through_binary64_pearson() {
    let source = include_str!("../src/kernels/precision.rs");
    for (begin, end) in [
        ("pub fn spearman_mixed", "pub fn spearman_f64"),
        ("pub fn spearman_f64", "pub fn mutual_info_f32"),
    ] {
        let (_, body) = source.split_once(begin).expect("Spearman profile function");
        let (body, _) = body.split_once(end).expect("Spearman profile function end");
        assert!(
            body.contains("pearson_f64(&x_ranks, &y_ranks)"),
            "{begin} must use the f64 Pearson SIMD/finalization route"
        );
    }
}
