#[test]
fn fp32_simd_implementation_has_no_wider_floating_arithmetic() {
    let source = include_str!("../src/simd/covariance_f32.rs");
    for forbidden in ["f64", "_pd", "cvtps_pd", "float64"] {
        assert!(
            !source.contains(forbidden),
            "fp32 SIMD source contains forbidden wider-lane token {forbidden:?}"
        );
    }

    assert!(source.contains("covariance_common::EqualVectorParts"));
    assert!(source.contains("isa::finite_dispatch_isa"));
    assert!(
        !source.contains("struct EqualVectorParts"),
        "fp32 SIMD must reuse the shared non-floating vector partition scaffold"
    );
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

#[test]
fn precision_fixed_mi_reuses_the_shared_f32_simd_histogram() {
    let source = include_str!("../src/kernels/precision.rs");
    let (_, production_body) = source
        .split_once("fn mutual_info_fixed_f32_with_scratch")
        .expect("precision fp32 fixed-bin production helper");
    let (production_body, _) = production_body
        .split_once("pub fn mutual_info_fixed_mixed")
        .expect("precision fp32 fixed-bin production helper end");
    let (_, fixture_body) = source
        .split_once("fn fixed_joint_f32")
        .expect("precision fp32 fixed-bin fixture helper");
    let (fixture_body, _) = fixture_body
        .split_once("fn fixed_bin_f32")
        .expect("precision fp32 fixed-bin fixture helper end");

    assert!(
        production_body.contains("crate::simd::fixed_bin_histogram2d"),
        "fp32/mixed fixed-bin MI must use the shared f32 SIMD histogram path"
    );
    assert!(
        !production_body.contains("fixed_bin_f32("),
        "fp32/mixed fixed-bin MI must not retain a duplicate scalar bin loop"
    );
    for forbidden in [
        "[0u32; MAX_FIXED_MI_BINS]",
        "[0u32; MAX_FIXED_MI_BINS * MAX_FIXED_MI_BINS]",
        ".to_vec()",
    ] {
        assert!(
            !fixture_body.contains(forbidden),
            "production fixed-bin MI must reuse worker scratch, not {forbidden:?}"
        );
    }
    assert!(
        source.contains("fixed_mi: FixedMiScratch"),
        "the production score scratch must retain fixed-bin buffers per Rayon worker"
    );
    assert!(
        production_body.contains("resize_fixed_histograms"),
        "the production fixed-bin path must resize and clear reusable buffers"
    );
}
