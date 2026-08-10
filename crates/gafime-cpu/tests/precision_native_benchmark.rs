//! Source gate for the standalone Core precision benchmark.
//!
//! Comparative evidence must never be produced by a Cargo test compiled from
//! the product checkout. The benchmark executable remains a standalone tracked
//! source under `tests/release_measure/`, compiled by the common-harness runner
//! against an explicit product `gafime-cpu` rlib. This integration target also
//! includes that exact source as a module so its methodology-only adversarial
//! unit tests execute in normal Rust validation without invoking benchmark
//! `main` or producing performance evidence.

#[allow(dead_code)]
#[path = "../../../tests/release_measure/core_precision_native_benchmark.rs"]
mod standalone_methodology_tests;

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
        "GAFIME_COMPILED_COMMAND_SHA256",
        "GAFIME_NATIVE_PRODUCT_RLIB_SHA256",
        "GAFIME_NATIVE_BENCH_BINARY_SHA256",
        "product_source_tree",
        "harness_source_tree",
        "order_position_median_ns",
        "confirmed_order_contamination_above_one_percent",
        "inconclusive_order_effect_requires_rerun",
        "whole_balanced_cycle_cluster",
        "BALANCED_SCHEDULE_CYCLES: usize = 5",
        "ORDER_MULTIPLE_COMPARISON_TESTS",
        "TARGET_REGION_NS: u128 = 100_000_000",
        "CALIBRATION_TARGET_REGION_NS: u128 = 200_000_000",
        "PER_SAMPLE_UNTIMED_PRECONDITION_MIN_NS",
        "precondition_duration_ns",
        "\\\"command_line\\\":",
        "MAX_LOOP_COUNT: usize = 4_096",
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
        "GAFIME_COMPILED_COMMAND_SHA256",
        "harness source differs from its checked-in HEAD blob",
    ] {
        assert!(
            STANDALONE_RUNNER.contains(marker),
            "Core common-harness runner is missing {marker}"
        );
    }
    assert!(
        STANDALONE_BENCHMARK.contains("old_three_of_five_direction_false_positive_is_inconclusive")
            && STANDALONE_BENCHMARK.contains("stable_repeated_position_effect_is_confirmed")
            && STANDALONE_BENCHMARK
                .contains("metric_ordinal_drift_does_not_alias_into_profile_position"),
        "the standalone source must retain its methodology adversarial tests"
    );
    assert!(
        !STANDALONE_RUNNER.contains("cargo test"),
        "the common harness must not compile a product-tree benchmark target"
    );
    assert!(
        !STANDALONE_BENCHMARK.contains("GAFIME_COMPILED_COMMAND_JSON"),
        "variable-length compiler commands must not affect benchmark code layout"
    );
}
