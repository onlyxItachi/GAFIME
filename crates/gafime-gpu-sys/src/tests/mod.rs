use super::*;
use crate::{
    abi::status_to_gpu_result,
    backend::{acquire_legacy_cuda_decision_path_lock, validate_decision_path_count},
};
use gafime_cpu::{
    decision_path::{path_membership, PathNode, SplitSign},
    matrix::CpuMatrix,
    CpuBackend,
};
use gafime_orchestrator::{
    config::EngineConfig,
    execute_plan,
    plan::combos::{build_continuous_plan, ContinuousPlanRequest, MI_TEMPLATE_BIN_LEVELS},
    prepare_continuous_execution, CompiledPlan, ComputeBackend, MatrixHandle,
};
use gafime_types::*;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Barrier, Mutex, MutexGuard,
};
use std::{env, ptr};

mod fixtures;
use fixtures::*;
mod cuda_continuous;
mod cuda_graph;
mod cuda_mi_spearman;
mod decision_path_rt;
mod loader_abi_lifecycle;
mod metal;
mod rocm;
