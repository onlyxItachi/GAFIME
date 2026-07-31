use super::*;
#[cfg(feature = "local-cmake-experiment")]
use crate::abi::status_to_gpu_result;
use gafime_cpu::{matrix::CpuMatrix, CpuBackend};
use gafime_orchestrator::{
    config::EngineConfig,
    execute_plan,
    plan::combos::{build_continuous_plan, ContinuousPlanRequest, MI_TEMPLATE_BIN_LEVELS},
    prepare_continuous_execution, CompiledPlan, ComputeBackend, MatrixHandle,
};
use gafime_types::*;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::{env, ptr};

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
