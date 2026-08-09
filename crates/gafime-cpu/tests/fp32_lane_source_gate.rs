#[test]
fn fp32_simd_implementation_has_no_wider_floating_arithmetic() {
    let source = include_str!("../src/simd/covariance_f32.rs");
    for forbidden in ["f64", "_pd", "cvtps_pd", "float64"] {
        assert!(
            !source.contains(forbidden),
            "fp32 SIMD source contains forbidden wider-lane token {forbidden:?}"
        );
    }
}

#[test]
fn public_fp32_pearson_is_only_a_binary32_simd_adapter() {
    let source = include_str!("../src/kernels/precision.rs");
    let start = source
        .find("pub fn pearson_f32")
        .expect("fp32 Pearson entrypoint");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("fp32 Pearson function end") + 3;
    let body = &body[..end];
    assert!(body.contains("crate::simd::pearson_corr_f32(x, y)"));
    for forbidden in ["f64", " as f64", "pearson_mixed", "pearson_f64"] {
        assert!(
            !body.contains(forbidden),
            "fp32 Pearson adapter contains forbidden token {forbidden:?}"
        );
    }
}
