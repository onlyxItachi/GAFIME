use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use gafime_orchestrator::MatrixHandle;
use gafime_types::{
    BackendKind, GafimeDecisionPathBatch, GafimeDecisionPathScoreBatch, GafimeDecisionPathTerm,
    GafimeGpuMatrix, GafimeResultTable, GafimeStatus, GAFIME_ABI_VERSION, GAFIME_BACKEND_CUDA,
    GAFIME_DECISION_PATH_FLAG_REQUIRE_RT, GAFIME_GPU_DEVICE_FLAG_OPTIX_RT,
    GAFIME_MAX_DECISION_PATH_COUNT, GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
use libloading::Library;

use crate::{
    abi::{load_optional_symbol, status_to_gpu_result, GpuFunctionTable, GpuSysError},
    backend::GpuBackend,
    profile::GpuDeviceProfile,
};

pub type GafimeGpuDecisionPathMembershipFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    paths: *const GafimeDecisionPathBatch,
) -> GafimeStatus;
pub type GafimeGpuDecisionPathScoreFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    paths: *const GafimeDecisionPathScoreBatch,
    result_out: *mut GafimeResultTable,
) -> GafimeStatus;
pub type GafimeGpuDecisionPathReleaseDeviceStateFn =
    unsafe extern "C" fn(device_id: u32) -> GafimeStatus;

#[derive(Clone, Copy, Default)]
pub struct LocalCmakeExperimentFunctions {
    pub decision_path_membership: Option<GafimeGpuDecisionPathMembershipFn>,
    pub decision_path_score: Option<GafimeGpuDecisionPathScoreFn>,
    pub decision_path_release_device_state: Option<GafimeGpuDecisionPathReleaseDeviceStateFn>,
}

/// # Safety
///
/// The library must be a trusted local CMake payload implementing the
/// experimental ABI declarations in src/cuda/rt_abi.hpp.
pub(crate) unsafe fn load_function_table(library: &Library) -> LocalCmakeExperimentFunctions {
    // SAFETY: the caller established the trusted payload boundary.
    unsafe {
        LocalCmakeExperimentFunctions {
            decision_path_membership: load_optional_symbol(
                library,
                "gafime_gpu_decision_path_membership",
            ),
            decision_path_score: load_optional_symbol(library, "gafime_gpu_decision_path_score"),
            decision_path_release_device_state: load_optional_symbol(
                library,
                "gafime_gpu_decision_path_release_device_state",
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecisionPathRtPolicy {
    #[default]
    AllowSmFallback,
    RequireRt,
}

impl DecisionPathRtPolicy {
    pub(crate) fn abi_flags(self) -> u32 {
        match self {
            Self::AllowSmFallback => 0,
            Self::RequireRt => GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        }
    }
}

impl GpuDeviceProfile {
    pub fn local_cmake_experiment_available(&self) -> bool {
        (self.flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LocalCmakeExperimentOwnerKey {
    payload_identity: usize,
    device_id: u32,
}

pub(crate) struct LocalCmakeExperimentDeviceStateOwner {
    key: LocalCmakeExperimentOwnerKey,
    device_id: u32,
    release: GafimeGpuDecisionPathReleaseDeviceStateFn,
    _library: Option<Arc<Library>>,
}

impl Drop for LocalCmakeExperimentDeviceStateOwner {
    fn drop(&mut self) {
        let mut owners = rt_device_state_owners()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let is_registered_owner = owners
            .get(&self.key)
            .is_some_and(|owner| std::ptr::eq(owner.as_ptr(), self));
        if !is_registered_owner {
            return;
        }
        owners.remove(&self.key);
        // SAFETY: the optional function came from the same trusted payload kept
        // alive by `_library`; the device id was validated when the backend was
        // constructed. Holding the registry lock prevents a replacement owner
        // from being installed until this final-owner cleanup completes. Drop
        // is best-effort because it cannot return an error.
        unsafe { (self.release)(self.device_id) };
    }
}

static RT_DEVICE_STATE_OWNERS: OnceLock<
    Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<LocalCmakeExperimentDeviceStateOwner>>>,
> = OnceLock::new();

fn rt_device_state_owners(
) -> &'static Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<LocalCmakeExperimentDeviceStateOwner>>>
{
    RT_DEVICE_STATE_OWNERS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn acquire_device_state_owner(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
    library: &Option<Arc<Library>>,
) -> Option<Arc<LocalCmakeExperimentDeviceStateOwner>> {
    if kind != GAFIME_BACKEND_CUDA {
        return None;
    }
    let release = functions
        .local_cmake_experiment
        .decision_path_release_device_state?;
    // The symbol address identifies the loaded payload even when callers reach
    // the same DSO through different hard-linked paths or loader Arc values.
    let payload_identity = release as usize;
    let key = LocalCmakeExperimentOwnerKey {
        payload_identity,
        device_id,
    };
    let mut owners = rt_device_state_owners()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(owner) = owners.get(&key).and_then(Weak::upgrade) {
        return Some(owner);
    }
    let owner = Arc::new(LocalCmakeExperimentDeviceStateOwner {
        key,
        device_id,
        release,
        _library: library.clone(),
    });
    owners.insert(key, Arc::downgrade(&owner));
    Some(owner)
}

static LEGACY_CUDA_DECISION_PATH_LOCKS: OnceLock<
    Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<Mutex<()>>>>,
> = OnceLock::new();

fn local_cmake_experiment_locks(
) -> &'static Mutex<HashMap<LocalCmakeExperimentOwnerKey, Weak<Mutex<()>>>> {
    LEGACY_CUDA_DECISION_PATH_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn acquire_local_cmake_experiment_lock(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
) -> Option<Arc<Mutex<()>>> {
    if kind != GAFIME_BACKEND_CUDA
        || functions
            .local_cmake_experiment
            .decision_path_release_device_state
            .is_some()
    {
        return None;
    }
    let payload_identity = functions
        .local_cmake_experiment
        .decision_path_score
        .map(|function| function as usize)
        .or_else(|| {
            functions
                .local_cmake_experiment
                .decision_path_membership
                .map(|function| function as usize)
        })?;
    let key = LocalCmakeExperimentOwnerKey {
        payload_identity,
        device_id,
    };
    let mut locks = local_cmake_experiment_locks()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(execution_lock) = locks.get(&key).and_then(Weak::upgrade) {
        return Some(execution_lock);
    }
    let execution_lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&execution_lock));
    Some(execution_lock)
}

pub(crate) fn validate_decision_path_count(path_count: usize) -> Result<(), GpuSysError> {
    if path_count > GAFIME_MAX_DECISION_PATH_COUNT as usize {
        Err(GpuSysError::SizeOverflow)
    } else {
        Ok(())
    }
}

impl GpuBackend {
    pub fn supports_decision_path_membership(&self) -> bool {
        self.functions
            .local_cmake_experiment
            .decision_path_membership
            .is_some()
    }

    pub fn supports_decision_path_score(&self) -> bool {
        self.functions
            .local_cmake_experiment
            .decision_path_score
            .is_some()
    }

    pub fn decision_path_membership(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        self.decision_path_membership_with_policy(
            matrix,
            terms,
            path_offsets,
            DecisionPathRtPolicy::AllowSmFallback,
        )
    }

    pub fn decision_path_membership_with_policy(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        policy: DecisionPathRtPolicy,
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        let Some(decision_path_membership) = self
            .functions
            .local_cmake_experiment
            .decision_path_membership
        else {
            return match policy {
                DecisionPathRtPolicy::AllowSmFallback => Ok(None),
                DecisionPathRtPolicy::RequireRt => Err(GpuSysError::BackendStatus {
                    operation: "gafime_gpu_decision_path_membership",
                    status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
                }),
            };
        };
        if policy == DecisionPathRtPolicy::RequireRt
            && (self.kind != GAFIME_BACKEND_CUDA
                || (self.device_flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) == 0)
        {
            return Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_membership",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            });
        }
        if matrix.backend_kind() != self.kind {
            return Err(GpuSysError::InvalidInput(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU decision-path membership requires a native resident matrix",
            ));
        }
        if terms.is_empty() || path_offsets.len() < 2 {
            return Err(GpuSysError::InvalidInput(
                "decision-path terms and offsets must be nonempty",
            ));
        }
        let path_count = path_offsets.len() - 1;
        validate_decision_path_count(path_count)?;
        if terms.len() > u32::MAX as usize {
            return Err(GpuSysError::SizeOverflow);
        }
        if path_offsets[0] != 0
            || path_offsets[path_count] as usize != terms.len()
            || path_offsets
                .windows(2)
                .any(|offsets| offsets[0] > offsets[1] || offsets[1] as usize > terms.len())
        {
            return Err(GpuSysError::InvalidInput(
                "decision-path offsets must be monotonic and cover exactly the terms buffer",
            ));
        }
        let rows = usize::try_from(matrix.rows()).map_err(|_| GpuSysError::SizeOverflow)?;
        let output_len = rows
            .checked_mul(path_count)
            .ok_or(GpuSysError::SizeOverflow)?;
        let mut membership = vec![f32::NAN; output_len];
        let batch = GafimeDecisionPathBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: path_count as u32,
            term_count: terms.len() as u32,
            flags: policy.abi_flags(),
            terms: terms.as_ptr(),
            path_offsets: path_offsets.as_ptr(),
            membership_host: membership.as_mut_ptr(),
            reserved: [0; 8],
        };
        let _legacy_execution_guard = self.local_cmake_experiment_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        // SAFETY: matrix identity and non-nullness, all slice lengths, monotonic
        // offsets, output size, and path-count bounds were checked above. The
        // slices and output Vec remain live for this synchronous payload call.
        let status = unsafe { decision_path_membership(matrix.raw(), &batch) };
        status_to_gpu_result("gafime_gpu_decision_path_membership", status)?;
        Ok(Some(membership))
    }

    pub fn decision_path_score(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        metric_ids: &[u32],
        result: &mut GafimeResultTable,
    ) -> Result<bool, GpuSysError> {
        self.decision_path_score_with_policy(
            matrix,
            terms,
            path_offsets,
            metric_ids,
            result,
            DecisionPathRtPolicy::AllowSmFallback,
        )
    }

    pub fn decision_path_score_with_policy(
        &mut self,
        matrix: &MatrixHandle,
        terms: &[GafimeDecisionPathTerm],
        path_offsets: &[u32],
        metric_ids: &[u32],
        result: &mut GafimeResultTable,
        policy: DecisionPathRtPolicy,
    ) -> Result<bool, GpuSysError> {
        let Some(decision_path_score) = self.functions.local_cmake_experiment.decision_path_score
        else {
            return match policy {
                DecisionPathRtPolicy::AllowSmFallback => Ok(false),
                DecisionPathRtPolicy::RequireRt => Err(GpuSysError::BackendStatus {
                    operation: "gafime_gpu_decision_path_score",
                    status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
                }),
            };
        };
        if policy == DecisionPathRtPolicy::RequireRt
            && (self.kind != GAFIME_BACKEND_CUDA
                || (self.device_flags & GAFIME_GPU_DEVICE_FLAG_OPTIX_RT) == 0)
        {
            return Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_score",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            });
        }
        if matrix.backend_kind() != self.kind {
            return Err(GpuSysError::InvalidInput(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU decision-path score requires a native resident matrix",
            ));
        }
        if terms.is_empty() || path_offsets.len() < 2 || metric_ids.is_empty() {
            return Err(GpuSysError::InvalidInput(
                "decision-path score terms, offsets, and metrics must be nonempty",
            ));
        }
        let path_count = path_offsets.len() - 1;
        validate_decision_path_count(path_count)?;
        if terms.len() > u32::MAX as usize || metric_ids.len() > u32::MAX as usize {
            return Err(GpuSysError::SizeOverflow);
        }
        if path_offsets[0] != 0
            || path_offsets[path_count] as usize != terms.len()
            || path_offsets
                .windows(2)
                .any(|offsets| offsets[0] > offsets[1] || offsets[1] as usize > terms.len())
        {
            return Err(GpuSysError::InvalidInput(
                "decision-path offsets must be monotonic and cover exactly the terms buffer",
            ));
        }
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: path_count as u32,
            term_count: terms.len() as u32,
            flags: policy.abi_flags(),
            terms: terms.as_ptr(),
            path_offsets: path_offsets.as_ptr(),
            metric_ids: metric_ids.as_ptr(),
            metric_count: metric_ids.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };
        let _legacy_execution_guard = self.local_cmake_experiment_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        // SAFETY: matrix identity/non-nullness and every batch slice/offset were
        // validated above. The caller-owned result table obeys the v1 ABI
        // allocation contract, and all borrowed storage remains live for this
        // synchronous call into the retained trusted payload.
        let status = unsafe { decision_path_score(matrix.raw(), &batch, result) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status)?;
        Ok(true)
    }
}
