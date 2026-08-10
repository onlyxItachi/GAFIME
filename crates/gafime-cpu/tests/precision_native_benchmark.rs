//! Source gate for the standalone Core precision benchmark.
//!
//! Comparative evidence must never be produced by a Cargo test compiled from
//! the product checkout.  The actual benchmark is a standalone tracked source
//! under `tests/release_measure/` and is compiled by the common-harness runner
//! against an explicit product `gafime-cpu` rlib.

const STANDALONE_BENCHMARK: &str =
    include_str!("../../../tests/release_measure/core_precision_native_benchmark.rs");
const STANDALONE_RUNNER: &str =
    include_str!("../../../tests/release_measure/run_core_precision_native_benchmark.py");

#[test]
fn core_release_benchmark_is_an_external_common_harness() {
    for marker in [
        "GAFIME_NATIVE_HARNESS_SOURCE",
        "GAFIME_NATIVE_HARNESS_SOURCE_SHA256",
        "GAFIME_NATIVE_HARNESS_SOURCE_GIT_BLOB",
        "GAFIME_COMPILED_HARNESS_RUNNER_SHA256",
        "GAFIME_NATIVE_PRODUCT_RLIB_SHA256",
        "GAFIME_NATIVE_BENCH_BINARY_SHA256",
        "product_source_tree",
        "harness_source_tree",
        "order_position_median_ns",
        "investigate_possible_order_contamination",
        "PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS",
        "precondition_duration_ns",
        "pearson_f32",
        "spearman_f32",
        "mutual_info_fixed_f32",
    ] {
        assert!(
            STANDALONE_BENCHMARK.contains(marker),
            "standalone Core benchmark is missing {marker}"
        );
    }
    for marker in [
        "--extern",
        "gafime_cpu=",
        "HARNESS_SOURCE",
        "_tracked_source_identity",
        "_compiler_environment",
        "harness source differs from its checked-in HEAD blob",
    ] {
        assert!(
            STANDALONE_RUNNER.contains(marker),
            "Core common-harness runner is missing {marker}"
        );
    }
    assert!(
        !STANDALONE_BENCHMARK.contains("#[test]"),
        "the release benchmark must not be a Cargo test"
    );
    assert!(
        !STANDALONE_RUNNER.contains("cargo test"),
        "the common harness must not compile a product-tree benchmark target"
    );
}
