use super::*;
#[cfg(feature = "local-cmake-experiment")]
use crate::abi::status_to_gpu_result;
use gafime_cpu::{matrix::CpuMatrix, precision::CpuPrecisionMatrix, CpuBackend};
use gafime_orchestrator::{
    config::EngineConfig,
    plan::combos::{build_continuous_plan, ContinuousPlanRequest, MI_TEMPLATE_BIN_LEVELS},
    prepare_continuous_execution, CompiledPlan, ComputeBackend, MatrixHandle,
};
use gafime_types::*;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::{env, ptr};

// Every invocation in this test module passes protocols owned by a live
// `CompiledPlan`/`PreparedContinuousExecution` and result descriptors rebound
// from a live `TestResultTable`. Keep the unsafe boundary visible without
// duplicating that invariant across every backend conformance test.
macro_rules! execute_plan {
    ($backend:expr, $matrix:expr, $plan:expr, $result:expr $(,)?) => {{
        // SAFETY: the test-owned plan and result table remain live for this
        // synchronous call and their descriptor lengths match their storage.
        unsafe { gafime_orchestrator::execute_plan($backend, $matrix, $plan, $result) }
    }};
}

mod fixtures;
use fixtures::*;
mod cuda_continuous;
mod cuda_graph;
mod cuda_mi_spearman;
mod loader_abi_lifecycle;
#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;
mod metal;
mod rocm;

#[test]
#[allow(clippy::type_complexity)]
fn raw_gpu_descriptor_routes_remain_unsafe_function_items() {
    #[allow(dead_code)]
    #[deny(unused_unsafe)]
    fn require_unsafe_calls(
        backend: &mut GpuBackend,
        matrix: &MatrixHandle,
        legacy: &GafimeLaunchProtocol,
        precision: &GafimePrecisionLaunchProtocol,
    ) {
        // SAFETY: compile-only API assertion; this function is never called.
        let _ = unsafe { backend.permutation_pvalues(matrix, legacy, &[], &[], 1) };
        // SAFETY: compile-only API assertion; this function is never called.
        let _ =
            unsafe { backend.permutation_pvalues_with_budget(matrix, legacy, &[], &[], 1, None) };
        // SAFETY: compile-only API assertion; this function is never called.
        let _ = unsafe {
            backend.permutation_pvalues_fp32_v2_with_budget(matrix, precision, &[], &[], 1, None)
        };
        // SAFETY: compile-only API assertion; this function is never called.
        let _ = unsafe {
            backend.permutation_pvalues_f64_v2_with_budget(matrix, precision, &[], &[], 1, None)
        };
    }

    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &GafimeLaunchProtocol,
        &[u64],
        &[f32],
        u32,
    ) -> Result<Option<Vec<f32>>, GpuSysError> = GpuBackend::permutation_pvalues;
    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &GafimeLaunchProtocol,
        &[u64],
        &[f32],
        u32,
        Option<u64>,
    ) -> Result<Option<Vec<f32>>, GpuSysError> = GpuBackend::permutation_pvalues_with_budget;
    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &GafimePrecisionLaunchProtocol,
        &[u64],
        &[f32],
        u32,
        Option<u64>,
    ) -> Result<Option<Vec<f32>>, GpuSysError> =
        GpuBackend::permutation_pvalues_fp32_v2_with_budget;
    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &GafimePrecisionLaunchProtocol,
        &[u64],
        &[f64],
        u32,
        Option<u64>,
    ) -> Result<Option<Vec<f64>>, GpuSysError> = GpuBackend::permutation_pvalues_f64_v2_with_budget;
}

#[cfg(feature = "local-cmake-experiment")]
#[test]
fn raw_local_experiment_result_routes_remain_unsafe_function_items() {
    #[allow(dead_code)]
    #[deny(unused_unsafe)]
    fn require_unsafe_calls(
        backend: &mut GpuBackend,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) {
        // SAFETY: compile-only API assertion; this function is never called.
        let _ = unsafe { backend.decision_path_score(matrix, &[], &[0], &[], result) };
        // SAFETY: compile-only API assertion; this function is never called.
        let _ = unsafe {
            backend.decision_path_score_with_policy(
                matrix,
                &[],
                &[0],
                &[],
                result,
                DecisionPathRtPolicy::AllowSmFallback,
            )
        };
    }

    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &[GafimeDecisionPathTerm],
        &[u32],
        &[u32],
        &mut GafimeResultTable,
    ) -> Result<bool, GpuSysError> = GpuBackend::decision_path_score;
    let _: unsafe fn(
        &mut GpuBackend,
        &MatrixHandle,
        &[GafimeDecisionPathTerm],
        &[u32],
        &[u32],
        &mut GafimeResultTable,
        DecisionPathRtPolicy,
    ) -> Result<bool, GpuSysError> = GpuBackend::decision_path_score_with_policy;
}
