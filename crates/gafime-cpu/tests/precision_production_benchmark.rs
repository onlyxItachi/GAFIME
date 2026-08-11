//! Methodology tests for the standalone Core production-executor benchmark.
//!
//! The release runner still compiles the tracked source against explicit
//! product rlibs. Including that exact source here only executes its pure
//! contract tests during normal Rust validation; benchmark `main` is never
//! invoked and no timing evidence is produced.

#[allow(
    dead_code,
    clippy::field_reassign_with_default,
    clippy::too_many_arguments
)]
#[path = "../../../tests/release_measure/core_precision_production_benchmark.rs"]
mod standalone_production_methodology_tests;
