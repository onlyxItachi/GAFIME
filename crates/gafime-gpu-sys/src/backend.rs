use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    BackendKind, GafimeDecisionPathBatch, GafimeDecisionPathScoreBatch, GafimeDecisionPathTerm,
    GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimeLaunchProtocol, GafimeMatrixDesc,
    GafimePermutationSignificanceTable, GafimeResultTable, GAFIME_ABI_VERSION, GAFIME_BACKEND_CUDA,
    GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_DTYPE_F32,
    GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_GPU_DEVICE_FLAG_OPTIX_RT, GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT, GAFIME_MATRIX_ROW_MAJOR,
    GAFIME_MAX_DECISION_PATH_COUNT, GAFIME_RESULT_FLAG_GRAPH_REPLAYED, GAFIME_STATUS_OK,
    GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
use libloading::Library;

use crate::{
    abi::{
        status_to_gpu_result, GafimeGpuDecisionPathReleaseDeviceStateFn, GpuFunctionTable,
        GpuSysError,
    },
    matrix::OwnedGpuMatrix,
    profile::{DecisionPathRtPolicy, GpuDeviceProfile},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RtDeviceStateOwnerKey {
    payload_identity: usize,
    device_id: u32,
}

pub(crate) struct RtDeviceStateOwner {
    key: RtDeviceStateOwnerKey,
    device_id: u32,
    release: GafimeGpuDecisionPathReleaseDeviceStateFn,
    _library: Option<Arc<Library>>,
}

impl Drop for RtDeviceStateOwner {
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
    Mutex<HashMap<RtDeviceStateOwnerKey, Weak<RtDeviceStateOwner>>>,
> = OnceLock::new();

fn rt_device_state_owners(
) -> &'static Mutex<HashMap<RtDeviceStateOwnerKey, Weak<RtDeviceStateOwner>>> {
    RT_DEVICE_STATE_OWNERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn acquire_rt_device_state_owner(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
    library: &Option<Arc<Library>>,
) -> Option<Arc<RtDeviceStateOwner>> {
    if kind != GAFIME_BACKEND_CUDA {
        return None;
    }
    let release = functions.decision_path_release_device_state?;
    // The symbol address identifies the loaded payload even when callers reach
    // the same DSO through different hard-linked paths or loader Arc values.
    let payload_identity = release as usize;
    let key = RtDeviceStateOwnerKey {
        payload_identity,
        device_id,
    };
    let mut owners = rt_device_state_owners()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(owner) = owners.get(&key).and_then(Weak::upgrade) {
        return Some(owner);
    }
    let owner = Arc::new(RtDeviceStateOwner {
        key,
        device_id,
        release,
        _library: library.clone(),
    });
    owners.insert(key, Arc::downgrade(&owner));
    Some(owner)
}

static LEGACY_CUDA_DECISION_PATH_LOCKS: OnceLock<
    Mutex<HashMap<RtDeviceStateOwnerKey, Weak<Mutex<()>>>>,
> = OnceLock::new();

fn legacy_cuda_decision_path_locks(
) -> &'static Mutex<HashMap<RtDeviceStateOwnerKey, Weak<Mutex<()>>>> {
    LEGACY_CUDA_DECISION_PATH_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn acquire_legacy_cuda_decision_path_lock(
    kind: BackendKind,
    device_id: u32,
    functions: &GpuFunctionTable,
) -> Option<Arc<Mutex<()>>> {
    if kind != GAFIME_BACKEND_CUDA || functions.decision_path_release_device_state.is_some() {
        return None;
    }
    let payload_identity = functions
        .decision_path_score
        .map(|function| function as usize)
        .or_else(|| {
            functions
                .decision_path_membership
                .map(|function| function as usize)
        })?;
    let key = RtDeviceStateOwnerKey {
        payload_identity,
        device_id,
    };
    let mut locks = legacy_cuda_decision_path_locks()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(execution_lock) = locks.get(&key).and_then(Weak::upgrade) {
        return Some(execution_lock);
    }
    let execution_lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&execution_lock));
    Some(execution_lock)
}

#[derive(Clone)]
pub struct GpuBackend {
    pub(crate) kind: BackendKind,
    pub(crate) device_id: u32,
    pub(crate) functions: GpuFunctionTable,
    pub(crate) device_flags: u32,
    pub(crate) library: Option<Arc<Library>>,
    pub(crate) library_path: Option<PathBuf>,
    pub(crate) legacy_cuda_decision_path_lock: Option<Arc<Mutex<()>>>,
}

pub(crate) fn validate_decision_path_count(path_count: usize) -> Result<(), GpuSysError> {
    if path_count > GAFIME_MAX_DECISION_PATH_COUNT as usize {
        Err(GpuSysError::SizeOverflow)
    } else {
        Ok(())
    }
}

impl GpuBackend {
    pub fn new(kind: BackendKind, functions: GpuFunctionTable) -> Result<Self, GpuSysError> {
        Self::from_function_table(kind, 0, functions, None, None)
    }

    pub(crate) fn from_function_table(
        kind: BackendKind,
        device_id: u32,
        functions: GpuFunctionTable,
        library: Option<Arc<Library>>,
        library_path: Option<PathBuf>,
    ) -> Result<Self, GpuSysError> {
        if !matches!(
            kind,
            GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
        ) {
            return Err(GpuSysError::InvalidInput(
                "GPU backend kind must be CUDA, ROCm, or Metal",
            ));
        }
        functions.require_complete()?;
        let legacy_cuda_decision_path_lock =
            acquire_legacy_cuda_decision_path_lock(kind, device_id, &functions);
        let mut backend = Self {
            kind,
            device_id,
            functions,
            device_flags: 0,
            library,
            library_path,
            legacy_cuda_decision_path_lock,
        };
        backend.device_flags = backend.device_info()?.flags;
        backend.graph_capability()?;
        Ok(backend)
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn loaded_library_path(&self) -> Option<&Path> {
        self.library_path.as_deref()
    }

    pub fn device_info(&self) -> Result<GafimeGpuDeviceInfo, GpuSysError> {
        let device_info = self
            .functions
            .device_info
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_device_info"))?;
        let mut info = GafimeGpuDeviceInfo::default();
        let status = unsafe { device_info(self.device_id, &mut info) };
        status_to_gpu_result("gafime_gpu_device_info", status)?;
        self.validate_device_identity(info.abi_version, info.backend_kind, info.device_id)?;
        Ok(info)
    }

    pub fn device_profile(&self) -> Result<GpuDeviceProfile, GpuSysError> {
        self.device_info()
            .map(|info| GpuDeviceProfile::from_info(&info))
    }

    pub fn graph_capability(&self) -> Result<GafimeGpuGraphCapability, GpuSysError> {
        let graph_capability = self
            .functions
            .graph_capability
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_graph_capability"))?;
        let mut capability = GafimeGpuGraphCapability::default();
        let status = unsafe { graph_capability(self.device_id, &mut capability) };
        status_to_gpu_result("gafime_gpu_graph_capability", status)?;
        self.validate_payload_identity(capability.abi_version, capability.backend_kind)?;
        Ok(capability)
    }

    fn validate_payload_identity(
        &self,
        abi_version: u32,
        backend_kind: BackendKind,
    ) -> Result<(), GpuSysError> {
        if abi_version != GAFIME_ABI_VERSION {
            return Err(GpuSysError::AbiVersionMismatch {
                expected: GAFIME_ABI_VERSION,
                actual: abi_version,
            });
        }
        if backend_kind != self.kind {
            return Err(GpuSysError::BackendKindMismatch {
                expected: self.kind,
                actual: backend_kind,
            });
        }
        Ok(())
    }

    fn validate_device_identity(
        &self,
        abi_version: u32,
        backend_kind: BackendKind,
        device_id: u32,
    ) -> Result<(), GpuSysError> {
        self.validate_payload_identity(abi_version, backend_kind)?;
        if device_id != self.device_id {
            return Err(GpuSysError::DeviceIdMismatch {
                expected: self.device_id,
                actual: device_id,
            });
        }
        Ok(())
    }

    pub fn supports_permutation_pvalues(&self) -> bool {
        self.functions.permutation_pvalues.is_some()
    }

    pub fn supports_permutation_memory_peak(&self) -> bool {
        self.functions.permutation_memory_peak.is_some()
    }

    pub fn supports_decision_path_membership(&self) -> bool {
        self.functions.decision_path_membership.is_some()
    }

    pub fn supports_decision_path_score(&self) -> bool {
        self.functions.decision_path_score.is_some()
    }

    /// Whether the loaded payload advertises the legacy immutable-protocol bit.
    /// Old ABI 1.0 payloads may return true without supporting generation tokens.
    pub fn supports_immutable_protocol(&self) -> bool {
        (self.device_flags & GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL) != 0
    }

    /// Whether immutable descriptors can be keyed by a caller-owned generation.
    pub fn supports_descriptor_generation(&self) -> bool {
        (self.device_flags & GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION) != 0
    }

    pub fn uses_fp64_mi_accumulation(&self) -> bool {
        (self.device_flags & gafime_types::GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64) != 0
    }

    pub fn supports_f64_storage(&self) -> bool {
        (self.device_flags & gafime_types::GAFIME_GPU_DEVICE_FLAG_F64_STORAGE) != 0
    }

    fn negotiate_launch_protocol(&self, protocol: &GafimeLaunchProtocol) -> GafimeLaunchProtocol {
        let mut negotiated = *protocol;
        if !self.supports_descriptor_generation() {
            negotiated.flags &= !GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
            negotiated.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0;
        }
        negotiated
    }

    pub fn alloc_matrix(&self, rows: u64, cols: u32) -> Result<OwnedGpuMatrix, GpuSysError> {
        if rows == 0 || cols == 0 {
            return Err(GpuSysError::InvalidInput(
                "matrix dimensions must be nonzero",
            ));
        }
        let element_count = rows
            .checked_mul(cols as u64)
            .ok_or(GpuSysError::SizeOverflow)?;
        let bytes = element_count
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or(GpuSysError::SizeOverflow)?;
        usize::try_from(bytes).map_err(|_| GpuSysError::SizeOverflow)?;
        let matrix_alloc = self
            .functions
            .matrix_alloc
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_matrix_alloc"))?;
        let desc = GafimeMatrixDesc {
            abi_version: GAFIME_ABI_VERSION,
            dtype: GAFIME_DTYPE_F32,
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            rows,
            cols,
            row_stride: cols,
            bytes,
        };
        // Acquire the payload/device lease before the native allocation. This
        // closes the race where the previous last matrix could release RT state
        // after a new matrix was allocated but before it received its lease.
        let rt_device_state_owner = acquire_rt_device_state_owner(
            self.kind,
            self.device_id,
            &self.functions,
            &self.library,
        );
        let mut raw = ptr::null_mut();
        let status = unsafe { matrix_alloc(self.device_id, &desc, &mut raw) };
        status_to_gpu_result("gafime_gpu_matrix_alloc", status)?;
        if raw.is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU ABI returned a null matrix handle",
            ));
        }
        Ok(OwnedGpuMatrix {
            // SAFETY: the payload returned a non-null matrix for this backend
            // and shape. OwnedGpuMatrix keeps both the free function and the
            // dynamic library alive and exposes the handle only by borrow.
            handle: unsafe { MatrixHandle::native(self.kind, raw, rows, cols) },
            functions: self.functions,
            library: self.library.clone(),
            _rt_device_state_owner: rt_device_state_owner,
        })
    }
}

impl GpuBackend {
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
        let Some(decision_path_membership) = self.functions.decision_path_membership else {
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
        let _legacy_execution_guard = self.legacy_cuda_decision_path_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
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
        let Some(decision_path_score) = self.functions.decision_path_score else {
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
        let _legacy_execution_guard = self.legacy_cuda_decision_path_lock.as_ref().map(|lock| {
            lock.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        let status = unsafe { decision_path_score(matrix.raw(), &batch, result) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status)?;
        Ok(true)
    }

    pub fn permutation_pvalues(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        candidate_ids: &[u64],
        observed_metric_values: &[f32],
        metric_count: u32,
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        self.permutation_pvalues_with_budget(
            matrix,
            protocol,
            candidate_ids,
            observed_metric_values,
            metric_count,
            None,
        )
    }

    pub fn permutation_pvalues_with_budget(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        candidate_ids: &[u64],
        observed_metric_values: &[f32],
        metric_count: u32,
        device_budget_bytes: Option<u64>,
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        let Some(permutation_pvalues) = self.functions.permutation_pvalues else {
            return Ok(None);
        };
        if matrix.backend_kind() != self.kind {
            return Err(GpuSysError::InvalidInput(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU permutation p-values require a native resident matrix",
            ));
        }
        if metric_count == 0 {
            return Err(GpuSysError::InvalidInput("metric_count must be nonzero"));
        }
        let expected = candidate_ids
            .len()
            .checked_mul(metric_count as usize)
            .ok_or(GpuSysError::SizeOverflow)?;
        if observed_metric_values.len() != expected {
            return Err(GpuSysError::InvalidInput(
                "observed metric grid length does not match rows*metrics",
            ));
        }
        let mut p_values = vec![f32::NAN; expected];
        let mut table = GafimePermutationSignificanceTable {
            abi_version: GAFIME_ABI_VERSION,
            metric_count,
            row_count: candidate_ids.len() as u64,
            candidate_ids: candidate_ids.as_ptr(),
            observed_metric_values: observed_metric_values.as_ptr(),
            p_values: p_values.as_mut_ptr(),
            reserved: [0; 8],
        };
        let negotiated_protocol = self.negotiate_launch_protocol(protocol);
        if let Some(device_budget_bytes) = device_budget_bytes {
            let Some(permutation_memory_peak) = self.functions.permutation_memory_peak else {
                // An older same-ABI payload may implement native p-values but
                // cannot prove that their retained allocation growth fits the
                // configured budget. Let the caller choose its budgeted path.
                return Ok(None);
            };
            let mut peak_bytes = 0u64;
            let status = unsafe {
                permutation_memory_peak(
                    matrix.raw(),
                    &negotiated_protocol,
                    candidate_ids.len() as u64,
                    &mut peak_bytes,
                )
            };
            status_to_gpu_result("gafime_gpu_permutation_memory_peak", status)?;
            if peak_bytes > device_budget_bytes {
                return Err(GpuSysError::InvalidInput(
                    "permutation p-value device-memory peak exceeds budget.vram_budget_mb",
                ));
            }
        }
        let status = unsafe { permutation_pvalues(matrix.raw(), &negotiated_protocol, &mut table) };
        status_to_gpu_result("gafime_gpu_permutation_pvalues", status)?;
        Ok(Some(p_values))
    }
}

impl ComputeBackend for GpuBackend {
    fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    fn execution_device_memory_peak_bytes(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        if matrix.backend_kind() != self.kind {
            return Err(OrchestratorError::InvalidPlan(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(OrchestratorError::InvalidPlan(
                "GPU backend requires a native resident matrix",
            ));
        }
        let Some(execution_memory_peak) = self.functions.execution_memory_peak else {
            return Ok(None);
        };
        let negotiated_protocol = self.negotiate_launch_protocol(protocol);
        let mut peak_bytes = 0u64;
        let status =
            unsafe { execution_memory_peak(matrix.raw(), &negotiated_protocol, &mut peak_bytes) };
        if status != GAFIME_STATUS_OK {
            return Err(OrchestratorError::BackendStatus(status));
        }
        Ok(Some(peak_bytes))
    }

    fn execute(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if matrix.backend_kind() != self.kind {
            return Err(OrchestratorError::InvalidPlan(
                "matrix backend does not match GPU backend",
            ));
        }
        if matrix.raw().is_null() {
            return Err(OrchestratorError::InvalidPlan(
                "GPU backend requires a native resident matrix",
            ));
        }
        let execute = self
            .functions
            .execute
            .ok_or(OrchestratorError::Unsupported(
                "GPU C ABI payload is not loaded",
            ))?;
        let negotiated_protocol = self.negotiate_launch_protocol(protocol);
        let status = unsafe { execute(matrix.raw(), &negotiated_protocol, result) };
        if status != GAFIME_STATUS_OK {
            return Err(OrchestratorError::BackendStatus(status));
        }
        Ok(BackendExecutionStats {
            launched_chunks: protocol.chunk_count as u64,
            graph_replays: u64::from((result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0),
            rows_written: result.row_count,
        })
    }
}
