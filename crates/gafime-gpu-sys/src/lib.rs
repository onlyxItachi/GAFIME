use std::{
    collections::HashMap,
    env,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex, OnceLock, Weak},
};

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    BackendKind, GafimeDecisionPathBatch, GafimeDecisionPathScoreBatch, GafimeDecisionPathTerm,
    GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimeGpuMatrix, GafimeLaunchProtocol,
    GafimeMatrixDesc, GafimePermutationSignificanceTable, GafimeResultTable, GafimeStatus,
    GAFIME_ABI_VERSION, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM,
    GAFIME_DECISION_PATH_FLAG_REQUIRE_RT, GAFIME_DTYPE_F32, GAFIME_GPU_ARCH_AMD_CDNA,
    GAFIME_GPU_ARCH_AMD_RDNA, GAFIME_GPU_ARCH_APPLE, GAFIME_GPU_ARCH_NVIDIA_ADA,
    GAFIME_GPU_ARCH_NVIDIA_AMPERE, GAFIME_GPU_ARCH_NVIDIA_BLACKWELL, GAFIME_GPU_ARCH_NVIDIA_HOPPER,
    GAFIME_GPU_ARCH_NVIDIA_TURING, GAFIME_GPU_ARCH_UNKNOWN, GAFIME_GPU_DEVICE_FLAG_AMD_CDNA,
    GAFIME_GPU_DEVICE_FLAG_AMD_RDNA, GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
    GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION, GAFIME_GPU_DEVICE_FLAG_DISCRETE,
    GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_GPU_DEVICE_FLAG_INTEGRATED, GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY,
    GAFIME_GPU_DEVICE_FLAG_OPTIX_RT, GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY,
    GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT,
    GAFIME_MATRIX_ROW_MAJOR, GAFIME_MAX_DECISION_PATH_COUNT, GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
    GAFIME_STATUS_OK, GAFIME_STATUS_UNSUPPORTED_BACKEND,
};
use libloading::Library;

pub const CUDA_LIBRARY_ENV: &str = "GAFIME_CUDA_V1_LIB";
pub const ROCM_LIBRARY_ENV: &str = "GAFIME_ROCM_V1_LIB";
pub const METAL_LIBRARY_ENV: &str = "GAFIME_METAL_V1_LIB";

type LibraryCacheKey = (BackendKind, PathBuf);

static GPU_LIBRARY_CACHE: OnceLock<Mutex<HashMap<LibraryCacheKey, Arc<Library>>>> = OnceLock::new();

fn library_cache() -> &'static Mutex<HashMap<LibraryCacheKey, Arc<Library>>> {
    GPU_LIBRARY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Keep immutable payload DSOs loaded for the process lifetime. This cache owns
/// no matrices, plans, results, or device buffers; cache-disabled eager calls
/// therefore remain one-shot executions without repeatedly paying loader and
/// native-runtime initialization costs.
unsafe fn load_process_library(
    path: &Path,
    kind: BackendKind,
) -> Result<Arc<Library>, GpuSysError> {
    let key_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let key = (kind, key_path);
    let mut cache = library_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(library) = cache.get(&key) {
        return Ok(library.clone());
    }
    let library = Arc::new(unsafe {
        Library::new(path).map_err(|err| GpuSysError::LoadLibrary {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?
    });
    cache.insert(key, library.clone());
    Ok(library)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuArchitectureClass {
    NvidiaTuring,
    NvidiaAmpere,
    NvidiaAda,
    NvidiaHopper,
    NvidiaBlackwell,
    AmdRdna,
    AmdCdna,
    Apple,
    VendorSpecific(u64),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuDeviceProfile {
    pub backend_kind: BackendKind,
    pub architecture: GpuArchitectureClass,
    pub flags: u32,
    pub unified_memory: bool,
    pub integrated: bool,
    pub discrete: bool,
    pub managed_memory: bool,
    pub high_bandwidth: bool,
    pub amd_rdna: bool,
    pub amd_cdna: bool,
    pub apple_family: bool,
    pub optix_rt: bool,
    pub immutable_protocol: bool,
    pub descriptor_generation: bool,
}

impl GpuDeviceProfile {
    pub fn from_info(info: &GafimeGpuDeviceInfo) -> Self {
        Self {
            backend_kind: info.backend_kind,
            architecture: architecture_class(info),
            flags: info.flags,
            unified_memory: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY),
            integrated: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_INTEGRATED),
            discrete: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_DISCRETE),
            managed_memory: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY),
            high_bandwidth: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH),
            amd_rdna: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_AMD_RDNA),
            amd_cdna: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_AMD_CDNA),
            apple_family: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY),
            optix_rt: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_OPTIX_RT),
            immutable_protocol: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL),
            descriptor_generation: has_device_flag(
                info,
                GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION,
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
    fn abi_flags(self) -> u32 {
        match self {
            Self::AllowSmFallback => 0,
            Self::RequireRt => GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        }
    }
}

pub fn architecture_class(info: &GafimeGpuDeviceInfo) -> GpuArchitectureClass {
    match info.reserved[0] {
        GAFIME_GPU_ARCH_NVIDIA_TURING => GpuArchitectureClass::NvidiaTuring,
        GAFIME_GPU_ARCH_NVIDIA_AMPERE => GpuArchitectureClass::NvidiaAmpere,
        GAFIME_GPU_ARCH_NVIDIA_ADA => GpuArchitectureClass::NvidiaAda,
        GAFIME_GPU_ARCH_NVIDIA_HOPPER => GpuArchitectureClass::NvidiaHopper,
        GAFIME_GPU_ARCH_NVIDIA_BLACKWELL => GpuArchitectureClass::NvidiaBlackwell,
        GAFIME_GPU_ARCH_AMD_RDNA => GpuArchitectureClass::AmdRdna,
        GAFIME_GPU_ARCH_AMD_CDNA => GpuArchitectureClass::AmdCdna,
        GAFIME_GPU_ARCH_APPLE => GpuArchitectureClass::Apple,
        GAFIME_GPU_ARCH_UNKNOWN => GpuArchitectureClass::Unknown,
        value => GpuArchitectureClass::VendorSpecific(value),
    }
}

pub fn has_device_flag(info: &GafimeGpuDeviceInfo, flag: u32) -> bool {
    (info.flags & flag) != 0
}

pub type GafimeGpuDeviceInfoFn =
    unsafe extern "C" fn(device_id: u32, info_out: *mut GafimeGpuDeviceInfo) -> GafimeStatus;
pub type GafimeGpuGraphCapabilityFn = unsafe extern "C" fn(
    device_id: u32,
    capability_out: *mut GafimeGpuGraphCapability,
) -> GafimeStatus;
pub type GafimeGpuMatrixAllocFn = unsafe extern "C" fn(
    device_id: u32,
    matrix_desc: *const GafimeMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus;
pub type GafimeGpuMatrixUploadFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    features_host: *const f32,
    target_host: *const f32,
    rows: u64,
    cols: u32,
) -> GafimeStatus;
pub type GafimeGpuMatrixUpdateTargetFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    target_host: *const f32,
    rows: u64,
) -> GafimeStatus;
pub type GafimeGpuMatrixFreeFn = unsafe extern "C" fn(matrix: GafimeGpuMatrix);
pub type GafimeGpuExecuteFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    result_out: *mut GafimeResultTable,
) -> GafimeStatus;
pub type GafimeGpuExecutionMemoryPeakFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    peak_bytes_out: *mut u64,
) -> GafimeStatus;
pub type GafimeGpuPermutationPvaluesFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    significance_out: *mut GafimePermutationSignificanceTable,
) -> GafimeStatus;
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

#[derive(Clone, Copy)]
pub struct GpuFunctionTable {
    pub device_info: Option<GafimeGpuDeviceInfoFn>,
    pub graph_capability: Option<GafimeGpuGraphCapabilityFn>,
    pub matrix_alloc: Option<GafimeGpuMatrixAllocFn>,
    pub matrix_upload: Option<GafimeGpuMatrixUploadFn>,
    pub matrix_update_target: Option<GafimeGpuMatrixUpdateTargetFn>,
    pub matrix_free: Option<GafimeGpuMatrixFreeFn>,
    pub execute: Option<GafimeGpuExecuteFn>,
    pub execution_memory_peak: Option<GafimeGpuExecutionMemoryPeakFn>,
    pub permutation_pvalues: Option<GafimeGpuPermutationPvaluesFn>,
    pub decision_path_membership: Option<GafimeGpuDecisionPathMembershipFn>,
    pub decision_path_score: Option<GafimeGpuDecisionPathScoreFn>,
    pub decision_path_release_device_state: Option<GafimeGpuDecisionPathReleaseDeviceStateFn>,
}

impl GpuFunctionTable {
    fn require_complete(&self) -> Result<(), GpuSysError> {
        if self.device_info.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_device_info"));
        }
        if self.graph_capability.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_graph_capability"));
        }
        if self.matrix_alloc.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_alloc"));
        }
        if self.matrix_upload.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_upload"));
        }
        if self.matrix_update_target.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_matrix_update_target",
            ));
        }
        if self.matrix_free.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_free"));
        }
        if self.execute.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_execute"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum GpuSysError {
    EnvMissing(&'static str),
    LoadLibrary {
        path: PathBuf,
        message: String,
    },
    LoadSymbol {
        symbol: &'static str,
        message: String,
    },
    MissingFunction(&'static str),
    InvalidInput(&'static str),
    AbiVersionMismatch {
        expected: u32,
        actual: u32,
    },
    BackendKindMismatch {
        expected: BackendKind,
        actual: BackendKind,
    },
    DeviceIdMismatch {
        expected: u32,
        actual: u32,
    },
    SizeOverflow,
    BackendStatus {
        operation: &'static str,
        status: GafimeStatus,
    },
}

impl fmt::Display for GpuSysError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvMissing(var) => write!(f, "{var} is not set"),
            Self::LoadLibrary { path, message } => {
                write!(f, "failed to load {}: {message}", path.display())
            }
            Self::LoadSymbol { symbol, message } => {
                write!(f, "failed to load symbol {symbol}: {message}")
            }
            Self::MissingFunction(symbol) => write!(f, "GPU ABI function {symbol} is missing"),
            Self::InvalidInput(message) => write!(f, "invalid GPU adapter input: {message}"),
            Self::AbiVersionMismatch { expected, actual } => write!(
                f,
                "GPU payload ABI version mismatch: expected {expected:#010x}, got {actual:#010x}"
            ),
            Self::BackendKindMismatch { expected, actual } => write!(
                f,
                "GPU payload backend mismatch: expected {expected}, got {actual}"
            ),
            Self::DeviceIdMismatch { expected, actual } => write!(
                f,
                "GPU payload device mismatch: expected {expected}, got {actual}"
            ),
            Self::SizeOverflow => write!(f, "GPU matrix size overflows u64 byte count"),
            Self::BackendStatus { operation, status } => {
                write!(f, "{operation} returned GPU ABI status {status}")
            }
        }
    }
}

impl Error for GpuSysError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RtDeviceStateOwnerKey {
    payload_identity: usize,
    device_id: u32,
}

struct RtDeviceStateOwner {
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

fn acquire_legacy_cuda_decision_path_lock(
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
    kind: BackendKind,
    device_id: u32,
    functions: GpuFunctionTable,
    device_flags: u32,
    library: Option<Arc<Library>>,
    library_path: Option<PathBuf>,
    legacy_cuda_decision_path_lock: Option<Arc<Mutex<()>>>,
}

fn validate_decision_path_count(path_count: usize) -> Result<(), GpuSysError> {
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

    fn from_function_table(
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

    pub fn cuda_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(CUDA_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(CUDA_LIBRARY_ENV))?;
        unsafe { Self::load_cuda_from_path(path, device_id) }
    }

    /// Load a ROCm/HIP payload implementing the same vendor-agnostic
    /// `gafime_gpu_*` C ABI as CUDA (the loader is vendor-generic; only the env
    /// var and `BackendKind` differ).
    pub fn rocm_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(ROCM_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(ROCM_LIBRARY_ENV))?;
        unsafe { Self::load_rocm_from_path(path, device_id) }
    }

    /// Load an Apple Metal payload implementing the same vendor-agnostic
    /// `gafime_gpu_*` C ABI as CUDA/ROCm.
    pub fn metal_from_env(device_id: u32) -> Result<Self, GpuSysError> {
        let path =
            env::var_os(METAL_LIBRARY_ENV).ok_or(GpuSysError::EnvMissing(METAL_LIBRARY_ENV))?;
        unsafe { Self::load_metal_from_path(path, device_id) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_cuda_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_CUDA) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_rocm_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_ROCM) }
    }

    /// # Safety
    ///
    /// `path` must name a trusted native library implementing the GAFIME GPU
    /// C ABI. Loading an untrusted or layout-incompatible dynamic library can
    /// execute arbitrary initialization code or expose invalid function tables.
    pub unsafe fn load_metal_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_METAL) }
    }

    unsafe fn load_abi_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
        kind: BackendKind,
    ) -> Result<Self, GpuSysError> {
        let path = path.as_ref().to_path_buf();
        let library = unsafe { load_process_library(&path, kind)? };
        let functions = unsafe { load_function_table(&library)? };
        Self::from_function_table(kind, device_id, functions, Some(library), Some(path))
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

pub struct OwnedGpuMatrix {
    handle: MatrixHandle,
    functions: GpuFunctionTable,
    library: Option<Arc<Library>>,
    _rt_device_state_owner: Option<Arc<RtDeviceStateOwner>>,
}

impl OwnedGpuMatrix {
    pub fn handle(&self) -> &MatrixHandle {
        &self.handle
    }

    pub fn rows(&self) -> u64 {
        self.handle.rows()
    }

    pub fn cols(&self) -> u32 {
        self.handle.cols()
    }

    pub fn upload(&self, features: &[f32], target: &[f32]) -> Result<(), GpuSysError> {
        let expected_features = self
            .rows()
            .checked_mul(self.cols() as u64)
            .ok_or(GpuSysError::SizeOverflow)?;
        let expected_features =
            usize::try_from(expected_features).map_err(|_| GpuSysError::SizeOverflow)?;
        if features.len() != expected_features {
            return Err(GpuSysError::InvalidInput(
                "feature buffer length does not match matrix dimensions",
            ));
        }
        let expected_target =
            usize::try_from(self.rows()).map_err(|_| GpuSysError::SizeOverflow)?;
        if target.len() != expected_target {
            return Err(GpuSysError::InvalidInput(
                "target buffer length does not match matrix rows",
            ));
        }
        let upload = self
            .functions
            .matrix_upload
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_matrix_upload"))?;
        let status = unsafe {
            upload(
                self.handle.raw(),
                features.as_ptr(),
                target.as_ptr(),
                self.rows(),
                self.cols(),
            )
        };
        status_to_gpu_result("gafime_gpu_matrix_upload", status)
    }

    pub fn update_target(&self, target: &[f32]) -> Result<(), GpuSysError> {
        let expected_target =
            usize::try_from(self.rows()).map_err(|_| GpuSysError::SizeOverflow)?;
        if target.len() != expected_target {
            return Err(GpuSysError::InvalidInput(
                "target buffer length does not match matrix rows",
            ));
        }
        let update_target =
            self.functions
                .matrix_update_target
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_matrix_update_target",
                ))?;
        let status = unsafe { update_target(self.handle.raw(), target.as_ptr(), self.rows()) };
        status_to_gpu_result("gafime_gpu_matrix_update_target", status)
    }
}

impl Drop for OwnedGpuMatrix {
    fn drop(&mut self) {
        if let Some(matrix_free) = self.functions.matrix_free {
            if !self.handle.raw().is_null() {
                unsafe { matrix_free(self.handle.raw()) };
            }
        }
        let _keep_library_alive = &self.library;
    }
}

unsafe fn load_function_table(library: &Library) -> Result<GpuFunctionTable, GpuSysError> {
    Ok(GpuFunctionTable {
        device_info: Some(unsafe {
            load_symbol::<GafimeGpuDeviceInfoFn>(library, "gafime_gpu_device_info")?
        }),
        graph_capability: Some(unsafe {
            load_symbol::<GafimeGpuGraphCapabilityFn>(library, "gafime_gpu_graph_capability")?
        }),
        matrix_alloc: Some(unsafe {
            load_symbol::<GafimeGpuMatrixAllocFn>(library, "gafime_gpu_matrix_alloc")?
        }),
        matrix_upload: Some(unsafe {
            load_symbol::<GafimeGpuMatrixUploadFn>(library, "gafime_gpu_matrix_upload")?
        }),
        matrix_update_target: Some(unsafe {
            load_symbol::<GafimeGpuMatrixUpdateTargetFn>(
                library,
                "gafime_gpu_matrix_update_target",
            )?
        }),
        matrix_free: Some(unsafe {
            load_symbol::<GafimeGpuMatrixFreeFn>(library, "gafime_gpu_matrix_free")?
        }),
        execute: Some(unsafe { load_symbol::<GafimeGpuExecuteFn>(library, "gafime_gpu_execute")? }),
        execution_memory_peak: unsafe {
            load_optional_symbol::<GafimeGpuExecutionMemoryPeakFn>(
                library,
                "gafime_gpu_execution_memory_peak",
            )
        },
        permutation_pvalues: unsafe {
            load_optional_symbol::<GafimeGpuPermutationPvaluesFn>(
                library,
                "gafime_gpu_permutation_pvalues",
            )
        },
        decision_path_membership: unsafe {
            load_optional_symbol::<GafimeGpuDecisionPathMembershipFn>(
                library,
                "gafime_gpu_decision_path_membership",
            )
        },
        decision_path_score: unsafe {
            load_optional_symbol::<GafimeGpuDecisionPathScoreFn>(
                library,
                "gafime_gpu_decision_path_score",
            )
        },
        decision_path_release_device_state: unsafe {
            load_optional_symbol::<GafimeGpuDecisionPathReleaseDeviceStateFn>(
                library,
                "gafime_gpu_decision_path_release_device_state",
            )
        },
    })
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &'static str) -> Result<T, GpuSysError> {
    let symbol_value = unsafe {
        library
            .get::<T>(format!("{symbol}\0").as_bytes())
            .map_err(|err| GpuSysError::LoadSymbol {
                symbol,
                message: err.to_string(),
            })?
    };
    Ok(*symbol_value)
}

unsafe fn load_optional_symbol<T: Copy>(library: &Library, symbol: &'static str) -> Option<T> {
    unsafe {
        library
            .get::<T>(format!("{symbol}\0").as_bytes())
            .ok()
            .map(|symbol_value| *symbol_value)
    }
}

fn status_to_gpu_result(operation: &'static str, status: GafimeStatus) -> Result<(), GpuSysError> {
    if status == GAFIME_STATUS_OK {
        Ok(())
    } else {
        Err(GpuSysError::BackendStatus { operation, status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_cpu::{
        decision_path::{path_membership, PathNode, SplitSign},
        matrix::CpuMatrix,
        CpuBackend,
    };
    use gafime_orchestrator::{
        config::EngineConfig,
        execute_plan,
        plan::combos::{build_continuous_plan, ContinuousPlanRequest, MI_TEMPLATE_BIN_LEVELS},
        prepare_continuous_execution, CompiledPlan,
    };
    use gafime_types::{
        GafimeDecisionPathTerm, GafimePermutationSchedule, GafimeRankSpec, GafimeResultTable,
        GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM,
        GAFIME_DECISION_PATH_FLAG_REQUIRE_RT, GAFIME_DECISION_PATH_SIGN_GT,
        GAFIME_DECISION_PATH_SIGN_LE, GAFIME_FAMILY_CONTINUOUS, GAFIME_FAMILY_DECISION_PATH,
        GAFIME_GPU_ARCH_AMD_CDNA, GAFIME_GPU_ARCH_APPLE, GAFIME_GPU_ARCH_NVIDIA_ADA,
        GAFIME_GPU_DEVICE_FLAG_AMD_CDNA, GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
        GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION, GAFIME_GPU_DEVICE_FLAG_DISCRETE,
        GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL,
        GAFIME_GPU_DEVICE_FLAG_INTEGRATED, GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY,
        GAFIME_GPU_DEVICE_FLAG_OPTIX_RT, GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY,
        GAFIME_GRAPH_STREAM_CAPTURE, GAFIME_LAUNCH_FLAG_GRAPH,
        GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT,
        GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
        GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
    };
    use std::sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Barrier, Mutex, MutexGuard,
    };

    static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());
    static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());
    static ABI_TEST_LOCK: Mutex<()> = Mutex::new(());
    static TEST_MATRIX_FREES: AtomicUsize = AtomicUsize::new(0);
    static TEST_EXECUTE_FLAGS: AtomicU32 = AtomicU32::new(0);
    static TEST_EXECUTE_DESCRIPTOR_GENERATION: AtomicU64 = AtomicU64::new(0);
    static TEST_DECISION_PATH_FLAGS: AtomicU32 = AtomicU32::new(0);
    static TEST_RT_RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);
    static TEST_RT_RELEASE_DEVICE_MASK: AtomicU32 = AtomicU32::new(0);

    unsafe extern "C" fn test_device_info(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        if info_out.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check above establishes a writable ABI output slot.
        unsafe {
            *info_out = GafimeGpuDeviceInfo {
                abi_version: GAFIME_ABI_VERSION,
                backend_kind: GAFIME_BACKEND_CUDA,
                device_id,
                ..Default::default()
            };
        }
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_device_info_wrong_abi(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).abi_version = GAFIME_ABI_VERSION + 1 };
        }
        status
    }

    unsafe extern "C" fn test_device_info_with_old_immutable_protocol(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).flags |= GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL };
        }
        status
    }

    unsafe extern "C" fn test_device_info_with_descriptor_generation(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info_with_old_immutable_protocol(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).flags |= GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION };
        }
        status
    }

    unsafe extern "C" fn test_device_info_with_optix_rt(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).flags |= GAFIME_GPU_DEVICE_FLAG_OPTIX_RT };
        }
        status
    }

    unsafe extern "C" fn test_device_info_wrong_backend(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).backend_kind = GAFIME_BACKEND_ROCM };
        }
        status
    }

    unsafe extern "C" fn test_device_info_wrong_device(
        device_id: u32,
        info_out: *mut GafimeGpuDeviceInfo,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_device_info(device_id, info_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*info_out).device_id = device_id.saturating_add(1) };
        }
        status
    }

    unsafe extern "C" fn test_graph_capability(
        _device_id: u32,
        capability_out: *mut GafimeGpuGraphCapability,
    ) -> GafimeStatus {
        if capability_out.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check above establishes a writable ABI output slot.
        unsafe {
            *capability_out = GafimeGpuGraphCapability {
                abi_version: GAFIME_ABI_VERSION,
                backend_kind: GAFIME_BACKEND_CUDA,
                ..Default::default()
            };
        }
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_graph_capability_wrong_backend(
        device_id: u32,
        capability_out: *mut GafimeGpuGraphCapability,
    ) -> GafimeStatus {
        // SAFETY: this stub forwards the caller's ABI arguments unchanged.
        let status = unsafe { test_graph_capability(device_id, capability_out) };
        if status == GAFIME_STATUS_OK {
            // SAFETY: the successful helper call initialized the output slot.
            unsafe { (*capability_out).backend_kind = GAFIME_BACKEND_METAL };
        }
        status
    }

    unsafe extern "C" fn test_matrix_alloc(
        _device_id: u32,
        _matrix_desc: *const GafimeMatrixDesc,
        matrix_out: *mut GafimeGpuMatrix,
    ) -> GafimeStatus {
        if matrix_out.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check establishes a writable output slot; the paired
        // test free function owns the allocation returned here.
        unsafe { *matrix_out = Box::into_raw(Box::new(0u8)).cast() };
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_matrix_upload(
        _matrix: GafimeGpuMatrix,
        _features_host: *const f32,
        _target_host: *const f32,
        _rows: u64,
        _cols: u32,
    ) -> GafimeStatus {
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_matrix_update_target(
        _matrix: GafimeGpuMatrix,
        _target_host: *const f32,
        _rows: u64,
    ) -> GafimeStatus {
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_matrix_free(matrix: GafimeGpuMatrix) {
        if !matrix.is_null() {
            // SAFETY: this function is paired only with test_matrix_alloc.
            unsafe { drop(Box::from_raw(matrix.cast::<u8>())) };
            TEST_MATRIX_FREES.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn test_execute(
        _matrix: GafimeGpuMatrix,
        _protocol: *const GafimeLaunchProtocol,
        _result_out: *mut GafimeResultTable,
    ) -> GafimeStatus {
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_execute_captures_launch_flags(
        _matrix: GafimeGpuMatrix,
        protocol: *const GafimeLaunchProtocol,
        _result_out: *mut GafimeResultTable,
    ) -> GafimeStatus {
        if protocol.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check establishes a readable launch protocol.
        let protocol = unsafe { &*protocol };
        TEST_EXECUTE_FLAGS.store(protocol.flags, Ordering::SeqCst);
        TEST_EXECUTE_DESCRIPTOR_GENERATION.store(
            protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT],
            Ordering::SeqCst,
        );
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_decision_path_membership_captures_flags(
        _matrix: GafimeGpuMatrix,
        paths: *const GafimeDecisionPathBatch,
    ) -> GafimeStatus {
        if paths.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check establishes a readable decision-path batch.
        TEST_DECISION_PATH_FLAGS.store(unsafe { (*paths).flags }, Ordering::SeqCst);
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_decision_path_score_captures_flags(
        _matrix: GafimeGpuMatrix,
        paths: *const GafimeDecisionPathScoreBatch,
        _result_out: *mut GafimeResultTable,
    ) -> GafimeStatus {
        if paths.is_null() {
            return gafime_types::GAFIME_STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: the null check establishes a readable decision-path batch.
        TEST_DECISION_PATH_FLAGS.store(unsafe { (*paths).flags }, Ordering::SeqCst);
        GAFIME_STATUS_OK
    }

    unsafe extern "C" fn test_release_decision_path_device_state(device_id: u32) -> GafimeStatus {
        TEST_RT_RELEASE_COUNT.fetch_add(1, Ordering::SeqCst);
        if device_id < u32::BITS {
            TEST_RT_RELEASE_DEVICE_MASK.fetch_or(1u32 << device_id, Ordering::SeqCst);
        }
        GAFIME_STATUS_OK
    }

    fn complete_test_function_table() -> GpuFunctionTable {
        GpuFunctionTable {
            device_info: Some(test_device_info),
            graph_capability: Some(test_graph_capability),
            matrix_alloc: Some(test_matrix_alloc),
            matrix_upload: Some(test_matrix_upload),
            matrix_update_target: Some(test_matrix_update_target),
            matrix_free: Some(test_matrix_free),
            execute: Some(test_execute),
            execution_memory_peak: None,
            permutation_pvalues: None,
            decision_path_membership: None,
            decision_path_score: None,
            decision_path_release_device_state: None,
        }
    }

    fn cuda_test_lock() -> MutexGuard<'static, ()> {
        CUDA_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn metal_test_lock() -> MutexGuard<'static, ()> {
        METAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn metal_backend_for_test() -> Option<GpuBackend> {
        env::var_os(METAL_LIBRARY_ENV)?;
        Some(
            GpuBackend::metal_from_env(0)
                .unwrap_or_else(|error| panic!("configured Metal payload failed to load: {error}")),
        )
    }

    fn cuda_backend_for_specialization_test() -> Option<GpuBackend> {
        env::var_os(CUDA_LIBRARY_ENV)?;
        Some(
            GpuBackend::cuda_from_env(0)
                .unwrap_or_else(|error| panic!("configured CUDA payload failed to load: {error}")),
        )
    }

    fn cuda_backend_with_optix_rt_for_test() -> Option<GpuBackend> {
        let backend = cuda_backend_for_specialization_test()?;
        let profile = backend
            .device_profile()
            .unwrap_or_else(|error| panic!("configured CUDA payload device query failed: {error}"));
        profile.optix_rt.then_some(backend)
    }

    fn rocm_backend_for_specialization_test() -> Option<GpuBackend> {
        env::var_os(ROCM_LIBRARY_ENV)?;
        Some(
            GpuBackend::rocm_from_env(0)
                .unwrap_or_else(|error| panic!("configured ROCm payload failed to load: {error}")),
        )
    }

    fn assert_configured_library_is_process_cached(env_name: &str, kind: BackendKind) {
        let Some(path) = env::var_os(env_name) else {
            return;
        };
        let first = unsafe { GpuBackend::load_abi_from_path(&path, 0, kind) }
            .unwrap_or_else(|error| panic!("configured payload failed to load: {error}"));
        let second = unsafe { GpuBackend::load_abi_from_path(&path, 0, kind) }
            .unwrap_or_else(|error| panic!("configured payload failed to reload: {error}"));
        let first_library = first.library.as_ref().expect("loaded payload owns its DSO");
        let second_library = second
            .library
            .as_ref()
            .expect("loaded payload owns its DSO");
        assert!(Arc::ptr_eq(first_library, second_library));
    }

    #[test]
    fn configured_payload_libraries_are_process_cached() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        assert_configured_library_is_process_cached(CUDA_LIBRARY_ENV, GAFIME_BACKEND_CUDA);
        assert_configured_library_is_process_cached(ROCM_LIBRARY_ENV, GAFIME_BACKEND_ROCM);
        assert_configured_library_is_process_cached(METAL_LIBRARY_ENV, GAFIME_BACKEND_METAL);
    }

    struct EnvVarOverride {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarOverride {
        fn set(key: &'static str, value: &'static str) -> Self {
            let previous = env::var_os(key);
            env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarOverride {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => env::set_var(self.key, value),
                None => env::remove_var(self.key),
            }
        }
    }

    struct TestResultTable {
        raw: GafimeResultTable,
        combo_indices: Vec<u32>,
        metric_values: Vec<f32>,
        ranks: Vec<u32>,
        families: Vec<u32>,
        candidate_ids: Vec<u64>,
        row_flags: Vec<u32>,
    }

    impl TestResultTable {
        fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
            let mut table = Self {
                raw: GafimeResultTable {
                    abi_version: GAFIME_ABI_VERSION,
                    max_arity,
                    metric_count,
                    flags: 0,
                    capacity,
                    row_count: 0,
                    combo_indices: ptr::null_mut(),
                    metric_values: ptr::null_mut(),
                    ranks: ptr::null_mut(),
                    families: ptr::null_mut(),
                    candidate_ids: ptr::null_mut(),
                    row_flags: ptr::null_mut(),
                    backend_private: ptr::null_mut(),
                    reserved: [0; 8],
                },
                combo_indices: vec![u32::MAX; capacity as usize * max_arity as usize],
                metric_values: vec![0.0; capacity as usize * metric_count as usize],
                ranks: vec![0; capacity as usize],
                families: vec![0; capacity as usize],
                candidate_ids: vec![0; capacity as usize],
                row_flags: vec![0; capacity as usize],
            };
            table.rebind();
            table
        }

        fn raw_mut(&mut self) -> &mut GafimeResultTable {
            self.rebind();
            &mut self.raw
        }

        fn metric_values(&self) -> &[f32] {
            &self.metric_values[..self.raw.row_count as usize * self.raw.metric_count as usize]
        }

        fn combo_indices(&self) -> &[u32] {
            &self.combo_indices[..self.raw.row_count as usize * self.raw.max_arity as usize]
        }

        fn ranks(&self) -> &[u32] {
            &self.ranks[..self.raw.row_count as usize]
        }

        fn candidate_ids(&self) -> &[u64] {
            &self.candidate_ids[..self.raw.row_count as usize]
        }

        fn rebind(&mut self) {
            self.raw.combo_indices = self.combo_indices.as_mut_ptr();
            self.raw.metric_values = self.metric_values.as_mut_ptr();
            self.raw.ranks = self.ranks.as_mut_ptr();
            self.raw.families = self.families.as_mut_ptr();
            self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
            self.raw.row_flags = self.row_flags.as_mut_ptr();
        }
    }

    #[test]
    fn gpu_backend_declares_vendor_kind() {
        let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
        assert_eq!(backend.backend_kind(), GAFIME_BACKEND_CUDA);
        assert!(!backend.supports_permutation_pvalues());
        assert!(!backend.supports_decision_path_membership());
        assert!(!backend.supports_decision_path_score());
        assert!(!backend.supports_immutable_protocol());
        assert!(!backend.supports_descriptor_generation());
        assert!(matches!(
            GpuBackend::new(GAFIME_BACKEND_CPU, complete_test_function_table()),
            Err(GpuSysError::InvalidInput(
                "GPU backend kind must be CUDA, ROCm, or Metal"
            ))
        ));
    }

    #[test]
    fn same_abi_payload_ranking_support_remains_probe_driven() {
        let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
        let capability = backend.graph_capability().unwrap();

        assert_eq!(capability.abi_version, GAFIME_ABI_VERSION);
        assert_eq!(capability.supports_device_ranking, 0);
    }

    #[test]
    fn descriptor_generation_is_sent_only_to_generation_capable_payloads() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON];
        config.budget.max_comb_size = 1;
        let prepared = prepare_continuous_execution(&config, 4, 2).unwrap();
        assert_eq!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );

        let execute_with = |device_info: GafimeGpuDeviceInfoFn| {
            let mut functions = complete_test_function_table();
            functions.device_info = Some(device_info);
            functions.execute = Some(test_execute_captures_launch_flags);
            let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
            let matrix = backend.alloc_matrix(4, 2).unwrap();
            let mut result = GafimeResultTable::default();
            TEST_EXECUTE_FLAGS.store(u32::MAX, Ordering::SeqCst);
            TEST_EXECUTE_DESCRIPTOR_GENERATION.store(u64::MAX, Ordering::SeqCst);
            prepared
                .execute(&mut backend, matrix.handle(), &mut result)
                .unwrap();
            (
                backend.supports_immutable_protocol(),
                backend.supports_descriptor_generation(),
                TEST_EXECUTE_FLAGS.load(Ordering::SeqCst),
                TEST_EXECUTE_DESCRIPTOR_GENERATION.load(Ordering::SeqCst),
            )
        };

        let (legacy_immutable, legacy_generation, legacy_flags, legacy_token) =
            execute_with(test_device_info);
        assert!(!legacy_immutable);
        assert!(!legacy_generation);
        assert_eq!(legacy_flags, prepared.plan().protocol().flags);
        assert_eq!(legacy_token, 0);

        let (old_immutable, old_generation, old_flags, old_token) =
            execute_with(test_device_info_with_old_immutable_protocol);
        assert!(old_immutable);
        assert!(!old_generation);
        assert_eq!(old_flags, prepared.plan().protocol().flags);
        assert_eq!(old_token, 0);

        let (current_immutable, current_generation, current_flags, current_token) =
            execute_with(test_device_info_with_descriptor_generation);
        assert!(current_immutable);
        assert!(current_generation);
        assert_eq!(
            current_flags,
            prepared.plan().protocol().flags | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL
        );
        assert_ne!(current_token, 0);
    }

    #[test]
    fn decision_path_count_reserves_the_terminal_offset_slot() {
        assert!(validate_decision_path_count(GAFIME_MAX_DECISION_PATH_COUNT as usize).is_ok());
        assert!(matches!(
            validate_decision_path_count(GAFIME_MAX_DECISION_PATH_COUNT as usize + 1),
            Err(GpuSysError::SizeOverflow)
        ));
    }

    #[test]
    fn require_rt_policy_rejects_an_unsupported_payload_in_rust() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut backend =
            GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
        let matrix = backend.alloc_matrix(2, 1).unwrap();
        let terms = [GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.5,
            ..Default::default()
        }];
        let offsets = [0, 1];

        assert!(backend
            .decision_path_membership(matrix.handle(), &terms, &offsets)
            .unwrap()
            .is_none());
        assert!(matches!(
            backend.decision_path_membership_with_policy(
                matrix.handle(),
                &terms,
                &offsets,
                DecisionPathRtPolicy::RequireRt,
            ),
            Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_membership",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            })
        ));

        let mut functions = complete_test_function_table();
        functions.decision_path_membership = Some(test_decision_path_membership_captures_flags);
        functions.decision_path_score = Some(test_decision_path_score_captures_flags);
        let mut sm_only_backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
        let sm_only_matrix = sm_only_backend.alloc_matrix(2, 1).unwrap();
        TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
        assert!(sm_only_backend
            .decision_path_membership(sm_only_matrix.handle(), &terms, &offsets)
            .unwrap()
            .is_some());
        assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), 0);
        TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
        assert!(matches!(
            sm_only_backend.decision_path_membership_with_policy(
                sm_only_matrix.handle(),
                &terms,
                &offsets,
                DecisionPathRtPolicy::RequireRt,
            ),
            Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_membership",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            })
        ));
        assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), u32::MAX);

        let mut result = GafimeResultTable::default();
        assert!(matches!(
            sm_only_backend.decision_path_score_with_policy(
                sm_only_matrix.handle(),
                &terms,
                &offsets,
                &[GAFIME_METRIC_PEARSON],
                &mut result,
                DecisionPathRtPolicy::RequireRt,
            ),
            Err(GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_score",
                status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
            })
        ));
        assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), u32::MAX);
    }

    #[test]
    fn rust_decision_path_policy_sets_the_approved_abi_flag() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut functions = complete_test_function_table();
        functions.device_info = Some(test_device_info_with_optix_rt);
        functions.decision_path_membership = Some(test_decision_path_membership_captures_flags);
        functions.decision_path_score = Some(test_decision_path_score_captures_flags);
        let mut backend = GpuBackend::new(GAFIME_BACKEND_CUDA, functions).unwrap();
        let matrix = backend.alloc_matrix(2, 1).unwrap();
        let terms = [GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.5,
            ..Default::default()
        }];
        let offsets = [0, 1];

        TEST_DECISION_PATH_FLAGS.store(u32::MAX, Ordering::SeqCst);
        backend
            .decision_path_membership(matrix.handle(), &terms, &offsets)
            .unwrap();
        assert_eq!(TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst), 0);

        backend
            .decision_path_membership_with_policy(
                matrix.handle(),
                &terms,
                &offsets,
                DecisionPathRtPolicy::RequireRt,
            )
            .unwrap();
        assert_eq!(
            TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst),
            GAFIME_DECISION_PATH_FLAG_REQUIRE_RT
        );

        let mut result = GafimeResultTable::default();
        backend
            .decision_path_score_with_policy(
                matrix.handle(),
                &terms,
                &offsets,
                &[GAFIME_METRIC_PEARSON],
                &mut result,
                DecisionPathRtPolicy::RequireRt,
            )
            .unwrap();
        assert_eq!(
            TEST_DECISION_PATH_FLAGS.load(Ordering::SeqCst),
            GAFIME_DECISION_PATH_FLAG_REQUIRE_RT
        );
    }

    #[test]
    fn legacy_cuda_decision_path_payloads_share_a_host_execution_lock() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut legacy_functions = complete_test_function_table();
        legacy_functions.decision_path_membership =
            Some(test_decision_path_membership_captures_flags);
        legacy_functions.decision_path_score = Some(test_decision_path_score_captures_flags);

        let first = GpuBackend::new(GAFIME_BACKEND_CUDA, legacy_functions).unwrap();
        let second = GpuBackend::new(GAFIME_BACKEND_CUDA, legacy_functions).unwrap();
        let first_lock = first
            .legacy_cuda_decision_path_lock
            .as_ref()
            .expect("pre-lifecycle CUDA payload needs host serialization");
        let second_lock = second
            .legacy_cuda_decision_path_lock
            .as_ref()
            .expect("same payload/device needs the shared host lock");
        assert!(Arc::ptr_eq(first_lock, second_lock));

        let mut current_functions = legacy_functions;
        current_functions.decision_path_release_device_state =
            Some(test_release_decision_path_device_state);
        let current = GpuBackend::new(GAFIME_BACKEND_CUDA, current_functions).unwrap();
        assert!(current.legacy_cuda_decision_path_lock.is_none());

        assert!(
            acquire_legacy_cuda_decision_path_lock(GAFIME_BACKEND_ROCM, 0, &legacy_functions,)
                .is_none()
        );
    }

    #[test]
    fn final_matrix_owner_releases_each_payload_device_state_once() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        TEST_RT_RELEASE_COUNT.store(0, Ordering::SeqCst);
        TEST_RT_RELEASE_DEVICE_MASK.store(0, Ordering::SeqCst);
        let mut functions = complete_test_function_table();
        functions.decision_path_release_device_state =
            Some(test_release_decision_path_device_state);
        let backend0 =
            GpuBackend::from_function_table(GAFIME_BACKEND_CUDA, 0, functions, None, None).unwrap();
        let backend1 =
            GpuBackend::from_function_table(GAFIME_BACKEND_CUDA, 1, functions, None, None).unwrap();

        let matrix0_first = backend0.alloc_matrix(2, 1).unwrap();
        let matrix0_second = backend0.alloc_matrix(2, 1).unwrap();
        let matrix1 = backend1.alloc_matrix(2, 1).unwrap();
        drop(matrix0_first);
        assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 0);
        drop(matrix1);
        assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(TEST_RT_RELEASE_DEVICE_MASK.load(Ordering::SeqCst), 0b10);
        drop(matrix0_second);
        assert_eq!(TEST_RT_RELEASE_COUNT.load(Ordering::SeqCst), 2);
        assert_eq!(TEST_RT_RELEASE_DEVICE_MASK.load(Ordering::SeqCst), 0b11);
    }

    #[test]
    fn gpu_backend_requires_every_mandatory_function() {
        macro_rules! assert_missing {
            ($field:ident, $symbol:literal) => {{
                let mut functions = complete_test_function_table();
                functions.$field = None;
                assert!(matches!(
                    GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
                    Err(GpuSysError::MissingFunction($symbol))
                ));
            }};
        }

        assert_missing!(device_info, "gafime_gpu_device_info");
        assert_missing!(graph_capability, "gafime_gpu_graph_capability");
        assert_missing!(matrix_alloc, "gafime_gpu_matrix_alloc");
        assert_missing!(matrix_upload, "gafime_gpu_matrix_upload");
        assert_missing!(matrix_update_target, "gafime_gpu_matrix_update_target");
        assert_missing!(matrix_free, "gafime_gpu_matrix_free");
        assert_missing!(execute, "gafime_gpu_execute");
    }

    #[test]
    fn gpu_backend_rejects_mismatched_payload_identity() {
        let mut functions = complete_test_function_table();
        functions.device_info = Some(test_device_info_wrong_abi);
        assert!(matches!(
            GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
            Err(GpuSysError::AbiVersionMismatch {
                expected: GAFIME_ABI_VERSION,
                actual,
            }) if actual == GAFIME_ABI_VERSION + 1
        ));

        let mut functions = complete_test_function_table();
        functions.device_info = Some(test_device_info_wrong_backend);
        assert!(matches!(
            GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
            Err(GpuSysError::BackendKindMismatch {
                expected: GAFIME_BACKEND_CUDA,
                actual: GAFIME_BACKEND_ROCM,
            })
        ));

        let mut functions = complete_test_function_table();
        functions.device_info = Some(test_device_info_wrong_device);
        assert!(matches!(
            GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
            Err(GpuSysError::DeviceIdMismatch {
                expected: 0,
                actual: 1,
            })
        ));

        let mut functions = complete_test_function_table();
        functions.graph_capability = Some(test_graph_capability_wrong_backend);
        assert!(matches!(
            GpuBackend::new(GAFIME_BACKEND_CUDA, functions),
            Err(GpuSysError::BackendKindMismatch {
                expected: GAFIME_BACKEND_CUDA,
                actual: GAFIME_BACKEND_METAL,
            })
        ));
    }

    #[test]
    fn owned_gpu_matrix_exposes_only_a_borrowed_handle_and_frees_once() {
        let _guard = ABI_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        TEST_MATRIX_FREES.store(0, Ordering::SeqCst);
        let backend = GpuBackend::new(GAFIME_BACKEND_CUDA, complete_test_function_table()).unwrap();
        let matrix = backend.alloc_matrix(2, 3).unwrap();
        let handle_fn: for<'a> fn(&'a OwnedGpuMatrix) -> &'a MatrixHandle = OwnedGpuMatrix::handle;
        {
            let handle = handle_fn(&matrix);
            assert_eq!(handle.backend_kind(), GAFIME_BACKEND_CUDA);
            assert_eq!((handle.rows(), handle.cols()), (2, 3));
            assert_eq!(TEST_MATRIX_FREES.load(Ordering::SeqCst), 0);
        }

        drop(matrix);
        assert_eq!(TEST_MATRIX_FREES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cuda_loader_requires_explicit_payload_path() {
        if env::var_os(CUDA_LIBRARY_ENV).is_some() {
            return;
        }
        assert!(matches!(
            GpuBackend::cuda_from_env(0),
            Err(GpuSysError::EnvMissing(CUDA_LIBRARY_ENV))
        ));
    }

    #[test]
    fn device_profile_interprets_portable_architecture_flags() {
        let mut cuda = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_CUDA,
            flags: GAFIME_GPU_DEVICE_FLAG_DISCRETE
                | GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH
                | GAFIME_GPU_DEVICE_FLAG_OPTIX_RT
                | GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL
                | GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION,
            reserved: [0; 8],
            ..Default::default()
        };
        cuda.reserved[0] = GAFIME_GPU_ARCH_NVIDIA_ADA;
        let profile = GpuDeviceProfile::from_info(&cuda);
        assert_eq!(profile.architecture, GpuArchitectureClass::NvidiaAda);
        assert!(profile.discrete);
        assert!(profile.high_bandwidth);
        assert!(profile.optix_rt);
        assert!(profile.immutable_protocol);
        assert!(profile.descriptor_generation);
        assert!(!profile.unified_memory);

        let mut rocm = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_ROCM,
            flags: GAFIME_GPU_DEVICE_FLAG_INTEGRATED
                | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
                | GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY
                | GAFIME_GPU_DEVICE_FLAG_AMD_CDNA,
            reserved: [0; 8],
            ..Default::default()
        };
        rocm.reserved[0] = GAFIME_GPU_ARCH_AMD_CDNA;
        let profile = GpuDeviceProfile::from_info(&rocm);
        assert_eq!(profile.architecture, GpuArchitectureClass::AmdCdna);
        assert!(profile.integrated);
        assert!(profile.unified_memory);
        assert!(profile.managed_memory);
        assert!(profile.amd_cdna);

        let mut metal = GafimeGpuDeviceInfo {
            backend_kind: GAFIME_BACKEND_METAL,
            flags: GAFIME_GPU_DEVICE_FLAG_INTEGRATED
                | GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY
                | GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY,
            reserved: [0; 8],
            ..Default::default()
        };
        metal.reserved[0] = GAFIME_GPU_ARCH_APPLE;
        let profile = GpuDeviceProfile::from_info(&metal);
        assert_eq!(profile.architecture, GpuArchitectureClass::Apple);
        assert!(profile.apple_family);
        assert!(profile.unified_memory);
    }

    #[test]
    fn cuda_device_profile_reports_runtime_architecture_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let info = backend.device_info().unwrap();
        let profile = GpuDeviceProfile::from_info(&info);
        assert_eq!(profile.backend_kind, GAFIME_BACKEND_CUDA);
        assert!(profile.discrete || profile.integrated);
        assert_ne!(profile.architecture, GpuArchitectureClass::Unknown);
        assert!(info.compute_major > 0);
        assert!(info.warp_size > 0);
        assert!(info.reserved[0] > 0);
    }

    #[test]
    fn cuda_adapter_executes_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let info = backend.device_info().unwrap();
        assert_eq!(info.backend_kind, GAFIME_BACKEND_CUDA);

        let rows = 4;
        let cols = 2;
        let features = vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
            return;
        };
        matrix.upload(&features, &target).unwrap();
        matrix.update_target(&target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );
        let mut result = TestResultTable::new(2, 1, 2);
        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(stats.launched_chunks, 1);
        assert_eq!(stats.rows_written, 2);
        assert_eq!(result.raw.row_count, 2);
        let values = result.metric_values();
        assert!((values[0] - 1.0).abs() < 1.0e-5);
        assert!((values[1] - 1.0).abs() < 1.0e-5);
        assert!((values[2] + 1.0).abs() < 1.0e-5);
        assert!((values[3] - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn cuda_cabi_rejects_stale_abi_overflow_and_malformed_inputs_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let matrix_alloc = backend.functions.matrix_alloc.unwrap();
        let mut raw = ptr::null_mut();
        let status = unsafe { matrix_alloc(0, ptr::null(), &mut raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);

        let stale_desc = GafimeMatrixDesc {
            abi_version: GAFIME_ABI_VERSION + 1,
            rows: 1,
            cols: 1,
            row_stride: 1,
            bytes: std::mem::size_of::<f32>() as u64,
            ..Default::default()
        };
        let status = unsafe { matrix_alloc(0, &stale_desc, &mut raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
        assert!(raw.is_null());

        let mismatched_bytes_desc = GafimeMatrixDesc {
            rows: 2,
            cols: 2,
            row_stride: 2,
            bytes: std::mem::size_of::<f32>() as u64,
            ..Default::default()
        };
        let status = unsafe { matrix_alloc(0, &mismatched_bytes_desc, &mut raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);
        assert!(raw.is_null());

        let huge_desc = GafimeMatrixDesc {
            rows: u64::MAX,
            cols: 2,
            row_stride: 2,
            bytes: u64::MAX,
            ..Default::default()
        };
        let status = unsafe { matrix_alloc(0, &huge_desc, &mut raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_OUT_OF_MEMORY);
        assert!(raw.is_null());

        let rows = 4u64;
        let cols = 2u32;
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON],
        );
        let term = GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 1.0,
            ..Default::default()
        };
        let offsets = [0u32, 1u32];
        let metric_ids = [GAFIME_METRIC_PEARSON];
        let mut membership = [0.0f32; 4];
        let stale_membership_batch = GafimeDecisionPathBatch {
            abi_version: GAFIME_ABI_VERSION + 1,
            path_count: 1,
            term_count: 1,
            flags: 0,
            terms: &term,
            path_offsets: offsets.as_ptr(),
            membership_host: membership.as_mut_ptr(),
            reserved: [0; 8],
        };
        let stale_score_batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION + 1,
            path_count: 1,
            term_count: 1,
            flags: 0,
            terms: &term,
            path_offsets: offsets.as_ptr(),
            metric_ids: metric_ids.as_ptr(),
            metric_count: 1,
            reserved32: 0,
            reserved: [0; 7],
        };
        let mut result = TestResultTable::new(2, 1, 1);
        let execute = backend.functions.execute.unwrap();
        let decision_path_membership = backend.functions.decision_path_membership.unwrap();
        let decision_path_score = backend.functions.decision_path_score.unwrap();

        let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), result.raw_mut()) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);
        let mut stale_protocol = *plan.protocol();
        stale_protocol.abi_version = GAFIME_ABI_VERSION + 1;
        let status = unsafe { execute(matrix.handle().raw(), &stale_protocol, result.raw_mut()) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
        result.raw_mut().abi_version = GAFIME_ABI_VERSION + 1;
        let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), &mut result.raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
        result.raw.abi_version = GAFIME_ABI_VERSION;
        let status =
            unsafe { decision_path_membership(matrix.handle().raw(), &stale_membership_batch) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
        let status = unsafe {
            decision_path_score(matrix.handle().raw(), &stale_score_batch, result.raw_mut())
        };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);

        matrix
            .upload(
                &[0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0],
                &[0.0, 1.0, 2.0, 3.0],
            )
            .unwrap();
        let mut malformed = *plan.protocol();
        let mut malformed_chunk = plan.chunks()[0];
        malformed_chunk.descriptor_count = 0;
        malformed.chunks = &malformed_chunk;
        let status = unsafe { execute(matrix.handle().raw(), &malformed, result.raw_mut()) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);

        let status = unsafe { execute(matrix.handle().raw(), &stale_protocol, result.raw_mut()) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);

        result.raw_mut().abi_version = GAFIME_ABI_VERSION + 1;
        let status = unsafe { execute(matrix.handle().raw(), plan.protocol(), &mut result.raw) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_ABI_MISMATCH);
        result.raw.abi_version = GAFIME_ABI_VERSION;

        let overflowing_batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: GAFIME_MAX_DECISION_PATH_COUNT + 1,
            term_count: 1,
            flags: 0,
            terms: &term,
            path_offsets: offsets.as_ptr(),
            metric_ids: metric_ids.as_ptr(),
            metric_count: 1,
            reserved32: 0,
            reserved: [0; 7],
        };
        let status = unsafe {
            decision_path_score(matrix.handle().raw(), &overflowing_batch, result.raw_mut())
        };
        assert_eq!(status, gafime_types::GAFIME_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn cuda_decision_path_membership_matches_cpu_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_membership() {
            return;
        }

        let rows = 5u64;
        let cols = 2u32;
        let features = vec![0.0, 0.0, 0.5, 0.6, 1.0, 1.0, f32::NAN, 1.0, 2.0, f32::NAN];
        let target = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 1, 3];
        let actual = backend
            .decision_path_membership(&matrix.handle(), &terms, &offsets)
            .unwrap()
            .expect("CUDA payload should expose decision-path membership");

        let columns = vec![0.0, 0.5, 1.0, f32::NAN, 2.0, 0.0, 0.6, 1.0, 1.0, f32::NAN];
        let expected0 = path_membership(
            &columns,
            rows as usize,
            &[PathNode {
                feature: 0,
                threshold: 0.5,
                sign: SplitSign::Le,
            }],
        );
        let expected1 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected = [expected0, expected1].concat();
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            if e.is_nan() {
                assert!(a.is_nan(), "membership[{idx}] expected NaN, got {a}");
            } else {
                assert_eq!(*a, *e, "membership[{idx}]");
            }
        }
    }

    #[test]
    fn cuda_require_rt_policy_matches_loaded_payload_capability_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_membership() {
            return;
        }
        let has_optix_rt = backend.device_profile().unwrap().optix_rt;
        let matrix = backend.alloc_matrix(4, 1).unwrap();
        matrix
            .upload(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0])
            .unwrap();
        let terms = [GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 1.0,
            ..Default::default()
        }];
        let result = backend.decision_path_membership_with_policy(
            matrix.handle(),
            &terms,
            &[0, 1],
            DecisionPathRtPolicy::RequireRt,
        );
        if has_optix_rt {
            assert_eq!(result.unwrap().unwrap(), [0.0, 0.0, 1.0, 1.0]);
        } else {
            assert!(matches!(
                result,
                Err(GpuSysError::BackendStatus {
                    operation: "gafime_gpu_decision_path_membership",
                    status: GAFIME_STATUS_UNSUPPORTED_BACKEND,
                })
            ));
        }
    }

    #[test]
    fn cuda_rt_same_device_concurrency_is_deterministic_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        if !backend.supports_decision_path_membership() {
            return;
        }
        drop(backend);

        const WORKERS: usize = 8;
        const REPEATS: usize = 20;
        let rows = 64u64;
        let cols = 2u32;
        let features = Arc::new(
            (0..rows)
                .flat_map(|row| {
                    let x = row as f32 / rows as f32;
                    [x, 1.0 - x]
                })
                .collect::<Vec<_>>(),
        );
        let target = Arc::new((0..rows).map(|row| row as f32).collect::<Vec<_>>());
        let expected = Arc::new(
            (0..rows)
                .map(|row| {
                    let x = row as f32 / rows as f32;
                    if x > 0.25 && 1.0 - x <= 0.75 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        );
        let barrier = Arc::new(Barrier::new(WORKERS));
        let mut workers = Vec::with_capacity(WORKERS);
        for _ in 0..WORKERS {
            let features = Arc::clone(&features);
            let target = Arc::clone(&target);
            let expected = Arc::clone(&expected);
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let mut backend = GpuBackend::cuda_from_env(0).unwrap();
                let matrix = backend.alloc_matrix(rows, cols).unwrap();
                matrix.upload(&features, &target).unwrap();
                let terms = [
                    GafimeDecisionPathTerm {
                        feature: 0,
                        sign: GAFIME_DECISION_PATH_SIGN_GT,
                        threshold: 0.25,
                        ..Default::default()
                    },
                    GafimeDecisionPathTerm {
                        feature: 1,
                        sign: GAFIME_DECISION_PATH_SIGN_LE,
                        threshold: 0.75,
                        ..Default::default()
                    },
                ];
                let offsets = [0u32, 2u32];
                barrier.wait();
                for _ in 0..REPEATS {
                    let actual = backend
                        .decision_path_membership_with_policy(
                            matrix.handle(),
                            &terms,
                            &offsets,
                            DecisionPathRtPolicy::RequireRt,
                        )
                        .unwrap()
                        .expect("RT-capable CUDA payload exposes membership");
                    assert_eq!(actual, *expected);
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn cuda_rt_explicit_cleanup_rebuilds_state_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Some(mut backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let Some(release_device_state) = backend.functions.decision_path_release_device_state
        else {
            return;
        };
        let matrix = backend.alloc_matrix(4, 1).unwrap();
        matrix
            .upload(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0])
            .unwrap();
        let terms = [GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 1.0,
            ..Default::default()
        }];
        let offsets = [0u32, 1u32];
        let first = backend
            .decision_path_membership_with_policy(
                matrix.handle(),
                &terms,
                &offsets,
                DecisionPathRtPolicy::RequireRt,
            )
            .unwrap()
            .unwrap();
        // SAFETY: the optional function belongs to the loaded payload, and the
        // backend's validated device id identifies the state being released.
        let status = unsafe { release_device_state(backend.device_id()) };
        status_to_gpu_result("gafime_gpu_decision_path_release_device_state", status).unwrap();
        let rebuilt = backend
            .decision_path_membership_with_policy(
                matrix.handle(),
                &terms,
                &offsets,
                DecisionPathRtPolicy::RequireRt,
            )
            .unwrap()
            .unwrap();
        assert_eq!(rebuilt, first);
    }

    #[test]
    fn cuda_decision_path_score_matches_cpu_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_score() {
            return;
        }

        let rows = 6u64;
        let cols = 2u32;
        let features = vec![0.0, 0.0, 0.4, 0.2, 0.6, 0.7, 1.0, 0.8, 1.4, 0.1, 2.0, 0.9];
        let target = vec![0.0, 0.2, 1.0, 1.4, 0.3, 2.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.8,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 2, 5];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let mut result = TestResultTable::new(2, 1, 2);
        let executed = backend
            .decision_path_score(
                &matrix.handle(),
                &terms,
                &offsets,
                &metrics,
                result.raw_mut(),
            )
            .unwrap();
        assert!(executed);
        assert_eq!(result.raw.row_count, 2);
        assert_eq!(result.combo_indices(), &[0, 1]);
        assert_eq!(result.candidate_ids(), &[0, 1]);
        assert_eq!(
            &result.families[..result.raw.row_count as usize],
            &[GAFIME_FAMILY_DECISION_PATH, GAFIME_FAMILY_DECISION_PATH]
        );

        let columns = vec![0.0, 0.4, 0.6, 1.0, 1.4, 2.0, 0.0, 0.2, 0.7, 0.8, 0.1, 0.9];
        let expected0 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected1 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 0,
                    threshold: 1.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.8,
                    sign: SplitSign::Le,
                },
            ],
        );
        let expected0_p = gafime_cpu::kernels::pearson(&expected0, &target);
        let expected1_p = gafime_cpu::kernels::pearson(&expected1, &target);
        let values = result.metric_values();
        assert!((values[0] - expected0_p).abs() < 1.0e-5);
        assert!((values[1] - expected0_p * expected0_p).abs() < 1.0e-5);
        assert!((values[2] - expected1_p).abs() < 1.0e-5);
        assert!((values[3] - expected1_p * expected1_p).abs() < 1.0e-5);
    }

    #[test]
    fn cuda_decision_path_direct_score_matches_cpu_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_score() {
            return;
        }

        let rows = 6u64;
        let cols = 2u32;
        let features = vec![0.0, 0.0, 0.4, 0.2, 0.6, 0.7, 1.0, 0.8, 1.4, 0.1, 2.0, 0.9];
        let target = vec![0.0, 0.2, 1.0, 1.4, 0.3, 2.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.8,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 2, 5];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let mut result = TestResultTable::new(2, 1, 2);
        let executed = backend
            .decision_path_score(
                &matrix.handle(),
                &terms,
                &offsets,
                &metrics,
                result.raw_mut(),
            )
            .unwrap();
        assert!(executed);
        assert_eq!(result.raw.row_count, 2);

        let columns = vec![0.0, 0.4, 0.6, 1.0, 1.4, 2.0, 0.0, 0.2, 0.7, 0.8, 0.1, 0.9];
        let expected0 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected1 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 0,
                    threshold: 1.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.8,
                    sign: SplitSign::Le,
                },
            ],
        );
        let expected0_p = gafime_cpu::kernels::pearson(&expected0, &target);
        let expected1_p = gafime_cpu::kernels::pearson(&expected1, &target);
        let values = result.metric_values();
        assert!((values[0] - expected0_p).abs() < 1.0e-4);
        assert!((values[1] - expected0_p * expected0_p).abs() < 1.0e-4);
        assert!((values[2] - expected1_p).abs() < 1.0e-4);
        assert!((values[3] - expected1_p * expected1_p).abs() < 1.0e-4);
    }

    #[test]
    fn cuda_decision_path_direct_score_groups_mixed_axes_when_rt_is_required() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 8u64;
        let cols = 4u32;
        let features = vec![
            0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0,
            0.5, 0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
        ];
        let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.4,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 2, 4, 6];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let mut result = TestResultTable::new(3, 1, 2);
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 3,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        assert_eq!(result.raw.row_count, 3);
        assert_eq!(result.combo_indices(), &[0, 1, 2]);
        assert_eq!(result.candidate_ids(), &[0, 1, 2]);

        let columns = vec![
            0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9,
            0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
        ];
        let expected0 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected1 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected2 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.4,
                    sign: SplitSign::Le,
                },
            ],
        );
        let values = result.metric_values();
        let expected = [
            gafime_cpu::kernels::pearson(&expected0, &target),
            gafime_cpu::kernels::pearson(&expected1, &target),
            gafime_cpu::kernels::pearson(&expected2, &target),
        ];
        for (path, pearson) in expected.iter().enumerate() {
            let base = path * 2;
            assert!(
                (values[base] - pearson).abs() < 1.0e-4,
                "path {path} pearson"
            );
            assert!(
                (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
                "path {path} r2"
            );
        }
    }

    #[test]
    fn cuda_decision_path_direct_score_groups_overlapping_pairs_when_rt_is_required() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 8u64;
        let cols = 4u32;
        let features = vec![
            0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0,
            0.5, 0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
        ];
        let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 2, 4, 6];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 3,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let columns = vec![
            0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9,
            0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
        ];
        let expected01 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected12 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected23 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );

        let mut result = TestResultTable::new(3, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        assert_eq!(result.raw.row_count, 3);
        let expected = [
            gafime_cpu::kernels::pearson(&expected01, &target),
            gafime_cpu::kernels::pearson(&expected12, &target),
            gafime_cpu::kernels::pearson(&expected23, &target),
        ];
        let values = result.metric_values();
        for (path, pearson) in expected.iter().enumerate() {
            let base = path * 2;
            assert!(
                (values[base] - pearson).abs() < 1.0e-4,
                "path {path} pearson"
            );
            assert!(
                (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
                "path {path} r2"
            );
        }
    }

    #[test]
    fn cuda_decision_path_direct_score_instanced_custom_aabbs_count_once() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 6u64;
        let cols = 4u32;
        let features = vec![
            0.25, 0.25, 1.50, 0.50, 0.50, 0.50, 0.25, 0.25, 0.75, 0.75, 0.50, 0.50, 1.50, 0.50,
            0.75, 0.75, 0.50, 1.50, 1.50, 0.50, -0.10, 0.50, 0.50, 1.50,
        ];
        let target = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 4, 8];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 2,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let columns = vec![
            0.25, 0.50, 0.75, 1.50, 0.50, -0.10, 0.25, 0.50, 0.75, 0.50, 1.50, 0.50, 1.50, 0.25,
            0.50, 0.75, 1.50, 0.50, 0.50, 0.25, 0.50, 0.75, 0.50, 1.50,
        ];
        let expected01 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.0,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 0,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.0,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
            ],
        );
        let expected23 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 2,
                    threshold: 0.0,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 2,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.0,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
            ],
        );
        let expected = [
            gafime_cpu::kernels::pearson(&expected01, &target),
            gafime_cpu::kernels::pearson(&expected23, &target),
        ];

        let mut result = TestResultTable::new(2, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        assert_eq!(result.raw.row_count, 2);
        let values = result.metric_values();
        for (path, pearson) in expected.iter().enumerate() {
            let base = path * 2;
            assert!(
                (values[base] - pearson).abs() < 1.0e-4,
                "path {path} pearson: got {}, expected {}",
                values[base],
                pearson
            );
            assert!(
                (values[base + 1] - pearson * pearson).abs() < 1.0e-4,
                "path {path} r2: got {}, expected {}",
                values[base + 1],
                pearson * pearson
            );
        }
    }

    #[test]
    fn cuda_decision_path_firsthit_score_partitioned_groups_match_cpu_when_rt_is_required() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 8u64;
        let cols = 4u32;
        let features = vec![
            0.2, 0.2, 0.2, 0.2, 0.7, 0.2, 0.7, 0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5,
            0.5, 0.5, 0.5, 0.5001, 0.5, 0.5001, 0.5, 0.5, 0.5001, 0.5, 0.5001, 0.9, 0.1, 0.1, 0.9,
        ];
        let target = vec![0.1, 1.0, 0.4, 1.4, 0.8, 1.2, 0.6, 0.3];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let mut terms = Vec::new();
        let mut offsets = vec![0u32];
        for &(feature0, feature1, sign0, sign1) in &[
            (
                0u32,
                1u32,
                GAFIME_DECISION_PATH_SIGN_LE,
                GAFIME_DECISION_PATH_SIGN_LE,
            ),
            (
                0u32,
                1u32,
                GAFIME_DECISION_PATH_SIGN_GT,
                GAFIME_DECISION_PATH_SIGN_LE,
            ),
            (
                2u32,
                3u32,
                GAFIME_DECISION_PATH_SIGN_LE,
                GAFIME_DECISION_PATH_SIGN_LE,
            ),
            (
                2u32,
                3u32,
                GAFIME_DECISION_PATH_SIGN_GT,
                GAFIME_DECISION_PATH_SIGN_LE,
            ),
        ] {
            terms.push(GafimeDecisionPathTerm {
                feature: feature0,
                sign: sign0,
                threshold: 0.5,
                ..Default::default()
            });
            if sign0 == GAFIME_DECISION_PATH_SIGN_GT {
                terms.push(GafimeDecisionPathTerm {
                    feature: feature0,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold: 1.0,
                    ..Default::default()
                });
            } else {
                terms.push(GafimeDecisionPathTerm {
                    feature: feature0,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: -0.1,
                    ..Default::default()
                });
            }
            terms.push(GafimeDecisionPathTerm {
                feature: feature1,
                sign: sign1,
                threshold: 0.5,
                ..Default::default()
            });
            terms.push(GafimeDecisionPathTerm {
                feature: feature1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: -0.1,
                ..Default::default()
            });
            offsets.push(terms.len() as u32);
        }
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 4,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let columns = vec![
            0.2, 0.7, 0.2, 0.7, 0.5, 0.5001, 0.5, 0.9, 0.2, 0.2, 0.7, 0.7, 0.5, 0.5, 0.5001, 0.1,
            0.2, 0.7, 0.2, 0.7, 0.5, 0.5001, 0.5, 0.1, 0.2, 0.2, 0.7, 0.7, 0.5, 0.5, 0.5001, 0.9,
        ];
        let expected_paths = [
            vec![
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 0,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
            ],
            vec![
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 0,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
            ],
            vec![
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 2,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 3,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
            ],
            vec![
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 2,
                    threshold: 1.0,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 3,
                    threshold: -0.1,
                    sign: SplitSign::Gt,
                },
            ],
        ];
        let mut result = TestResultTable::new(4, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        assert_eq!(result.raw.row_count, 4);
        assert_eq!(result.combo_indices(), &[0, 1, 2, 3]);

        let values = result.metric_values();
        for (path, nodes) in expected_paths.iter().enumerate() {
            let membership = path_membership(&columns, rows as usize, nodes);
            let pearson = gafime_cpu::kernels::pearson(&membership, &target);
            let base = path * 2;
            assert!(
                (values[base] - pearson).abs() < 1.0e-5,
                "path {path} pearson: got {}, expected {}",
                values[base],
                pearson
            );
            assert!(
                (values[base + 1] - pearson * pearson).abs() < 1.0e-5,
                "path {path} r2: got {}, expected {}",
                values[base + 1],
                pearson * pearson
            );
        }
    }

    #[test]
    fn cuda_decision_path_tiny_bounded_regions_respect_rt_numeric_domain() {
        let _cuda_guard = cuda_test_lock();
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");
        let rows = 4u64;
        let cols = 2u32;
        let target = vec![1.0, 0.0, 0.0, 0.0];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let offsets = vec![0u32, 4];

        for (threshold, rt_representable) in [(f32::from_bits(1), false), (f32::MIN_POSITIVE, true)]
        {
            let features = vec![
                threshold, threshold, 0.0, threshold, threshold, 0.0, 0.0, 0.0,
            ];
            let matrix = backend.alloc_matrix(rows, cols).unwrap();
            matrix.upload(&features, &target).unwrap();
            let terms = vec![
                GafimeDecisionPathTerm {
                    feature: 0,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: 0.0,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 0,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 1,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: 0.0,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 1,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold,
                    ..Default::default()
                },
            ];

            {
                let _score_mode =
                    EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
                let required_batch = GafimeDecisionPathScoreBatch {
                    abi_version: GAFIME_ABI_VERSION,
                    path_count: 1,
                    term_count: terms.len() as u32,
                    flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
                    terms: terms.as_ptr(),
                    path_offsets: offsets.as_ptr(),
                    metric_ids: metrics.as_ptr(),
                    metric_count: metrics.len() as u32,
                    reserved32: 0,
                    reserved: [0; 7],
                };
                let mut result = TestResultTable::new(1, 1, 2);
                let status = unsafe {
                    decision_path_score(matrix.handle().raw(), &required_batch, result.raw_mut())
                };
                if rt_representable {
                    status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
                    assert_eq!(result.metric_values(), &[1.0, 1.0]);
                } else {
                    assert_eq!(status, GAFIME_STATUS_UNSUPPORTED_BACKEND);
                }
            }

            {
                let _score_mode =
                    EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
                let fallback_batch = GafimeDecisionPathScoreBatch {
                    abi_version: GAFIME_ABI_VERSION,
                    path_count: 1,
                    term_count: terms.len() as u32,
                    flags: 0,
                    terms: terms.as_ptr(),
                    path_offsets: offsets.as_ptr(),
                    metric_ids: metrics.as_ptr(),
                    metric_count: metrics.len() as u32,
                    reserved32: 0,
                    reserved: [0; 7],
                };
                let mut result = TestResultTable::new(1, 1, 2);
                let status = unsafe {
                    decision_path_score(matrix.handle().raw(), &fallback_batch, result.raw_mut())
                };
                status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
                assert_eq!(result.metric_values(), &[1.0, 1.0]);
            }
        }
    }

    #[test]
    fn cuda_decision_path_firsthit_bucket_lattice_covers_narrow_float_boundaries() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");
        let cutoff = 2.0_f32.powi(-60);
        for threshold in [
            f32::from_bits(cutoff.to_bits() - 1),
            cutoff,
            f32::from_bits(cutoff.to_bits() + 1),
        ] {
            let above_threshold = f32::from_bits(threshold.to_bits() + 1);
            let min_normal = f32::MIN_POSITIVE;
            let rows = 5u64;
            let cols = 2u32;
            let features = vec![
                min_normal,
                min_normal,
                threshold,
                threshold,
                0.0,
                min_normal,
                min_normal,
                0.0,
                above_threshold,
                above_threshold,
            ];
            let target = vec![1.0, 1.0, 0.0, 0.0, 0.0];
            let matrix = backend.alloc_matrix(rows, cols).unwrap();
            matrix.upload(&features, &target).unwrap();
            let terms = vec![
                GafimeDecisionPathTerm {
                    feature: 0,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: 0.0,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 0,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 1,
                    sign: GAFIME_DECISION_PATH_SIGN_GT,
                    threshold: 0.0,
                    ..Default::default()
                },
                GafimeDecisionPathTerm {
                    feature: 1,
                    sign: GAFIME_DECISION_PATH_SIGN_LE,
                    threshold,
                    ..Default::default()
                },
            ];
            let offsets = vec![0u32, terms.len() as u32];
            let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
            let batch = GafimeDecisionPathScoreBatch {
                abi_version: GAFIME_ABI_VERSION,
                path_count: 1,
                term_count: terms.len() as u32,
                flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
                terms: terms.as_ptr(),
                path_offsets: offsets.as_ptr(),
                metric_ids: metrics.as_ptr(),
                metric_count: metrics.len() as u32,
                reserved32: 0,
                reserved: [0; 7],
            };
            let mut result = TestResultTable::new(1, 1, 2);

            let status =
                unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };

            status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
            assert_eq!(result.metric_values(), &[1.0, 1.0]);
        }
    }

    #[test]
    fn cuda_decision_path_firsthit_score_rejects_overlap_without_sm_fallback() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "firsthit");
        let Ok(backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let Some(decision_path_score) = backend.functions.decision_path_score else {
            return;
        };

        let rows = 4u64;
        let cols = 2u32;
        let features = vec![0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.4, 0.7];
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.75,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.75,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.25,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.25,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 1.0,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 4, 8];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 2,
            term_count: terms.len() as u32,
            flags: 0,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let mut result = TestResultTable::new(2, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result.raw_mut()) };
        assert_eq!(status, gafime_types::GAFIME_STATUS_UNSUPPORTED_BACKEND);
    }

    #[test]
    fn cuda_decision_path_direct_score_recomputes_target_stats_with_cached_points() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 8u64;
        let cols = 4u32;
        let features = vec![
            0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0,
            0.5, 0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
        ];
        let target0 = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
        let target1 = vec![1.6, 0.1, 0.4, 1.9, 0.3, 1.4, 0.2, 1.1];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target0).unwrap();

        let terms = vec![
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
        ];
        let offsets = vec![0u32, 2, 4];
        let metrics = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        let batch = GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 2,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let columns = vec![
            0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9,
            0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
        ];
        let expected0 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected1 = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );

        let mut result0 = TestResultTable::new(2, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result0.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();
        matrix.update_target(&target1).unwrap();
        let mut result1 = TestResultTable::new(2, 1, 2);
        let status =
            unsafe { decision_path_score(matrix.handle().raw(), &batch, result1.raw_mut()) };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

        let expected_first = [
            gafime_cpu::kernels::pearson(&expected0, &target0),
            gafime_cpu::kernels::pearson(&expected1, &target0),
        ];
        let expected_second = [
            gafime_cpu::kernels::pearson(&expected0, &target1),
            gafime_cpu::kernels::pearson(&expected1, &target1),
        ];
        let values0 = result0.metric_values();
        let values1 = result1.metric_values();
        assert!((values0[0] - expected_first[0]).abs() < 1.0e-4);
        assert!((values0[2] - expected_first[1]).abs() < 1.0e-4);
        assert!((values1[0] - expected_second[0]).abs() < 1.0e-4);
        assert!((values1[2] - expected_second[1]).abs() < 1.0e-4);
        assert!(
            (values0[0] - values1[0]).abs() > 1.0e-3 || (values0[2] - values1[2]).abs() > 1.0e-3,
            "target-only update must change direct RT scores while reusing packed points"
        );
    }

    #[test]
    fn cuda_decision_path_direct_score_refreshes_cached_scatter_map() {
        let _cuda_guard = cuda_test_lock();
        let _score_mode = EnvVarOverride::set("GAFIME_CUDA_DECISION_PATH_RT_SCORE", "direct");
        let Some(backend) = cuda_backend_with_optix_rt_for_test() else {
            return;
        };
        let decision_path_score = backend
            .functions
            .decision_path_score
            .expect("OptiX CUDA payload must expose decision-path scoring");

        let rows = 8u64;
        let cols = 4u32;
        let features = vec![
            0.1, 0.1, 0.9, 0.2, 0.6, 0.7, 0.1, 0.8, 0.8, 0.2, 0.7, 0.6, 0.3, 0.9, 0.4, 0.4, 1.0,
            0.5, 0.8, 0.9, 0.2, 0.4, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7,
        ];
        let target = vec![0.1, 1.3, 1.1, 0.6, 1.7, 0.2, 1.2, 0.9];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let path01_gt = [
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
        ];
        let path23_gt = [
            GafimeDecisionPathTerm {
                feature: 2,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 3,
                sign: GAFIME_DECISION_PATH_SIGN_GT,
                threshold: 0.5,
                ..Default::default()
            },
        ];
        let path01_le = [
            GafimeDecisionPathTerm {
                feature: 0,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.5,
                ..Default::default()
            },
            GafimeDecisionPathTerm {
                feature: 1,
                sign: GAFIME_DECISION_PATH_SIGN_LE,
                threshold: 0.4,
                ..Default::default()
            },
        ];
        let terms_first = [path01_gt, path23_gt, path01_le].concat();
        let terms_second = [path01_gt, path01_le, path23_gt].concat();
        let offsets = vec![0u32, 2, 4, 6];
        let metrics = vec![GAFIME_METRIC_PEARSON];
        let make_batch = |terms: &[GafimeDecisionPathTerm]| GafimeDecisionPathScoreBatch {
            abi_version: GAFIME_ABI_VERSION,
            path_count: 3,
            term_count: terms.len() as u32,
            flags: GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
            terms: terms.as_ptr(),
            path_offsets: offsets.as_ptr(),
            metric_ids: metrics.as_ptr(),
            metric_count: metrics.len() as u32,
            reserved32: 0,
            reserved: [0; 7],
        };

        let columns = vec![
            0.1, 0.6, 0.8, 0.3, 1.0, 0.2, 0.7, 0.4, 0.1, 0.7, 0.2, 0.9, 0.5, 0.4, 0.8, 0.6, 0.9,
            0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1, 0.3, 0.7,
        ];
        let expected01_gt = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected23_gt = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 2,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
                PathNode {
                    feature: 3,
                    threshold: 0.5,
                    sign: SplitSign::Gt,
                },
            ],
        );
        let expected01_le = path_membership(
            &columns,
            rows as usize,
            &[
                PathNode {
                    feature: 0,
                    threshold: 0.5,
                    sign: SplitSign::Le,
                },
                PathNode {
                    feature: 1,
                    threshold: 0.4,
                    sign: SplitSign::Le,
                },
            ],
        );
        let expected = [
            gafime_cpu::kernels::pearson(&expected01_gt, &target),
            gafime_cpu::kernels::pearson(&expected23_gt, &target),
            gafime_cpu::kernels::pearson(&expected01_le, &target),
        ];

        let mut result_first = TestResultTable::new(3, 1, 1);
        let batch_first = make_batch(&terms_first);
        let status = unsafe {
            decision_path_score(matrix.handle().raw(), &batch_first, result_first.raw_mut())
        };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

        let mut result_second = TestResultTable::new(3, 1, 1);
        let batch_second = make_batch(&terms_second);
        let status = unsafe {
            decision_path_score(
                matrix.handle().raw(),
                &batch_second,
                result_second.raw_mut(),
            )
        };
        status_to_gpu_result("gafime_gpu_decision_path_score", status).unwrap();

        let first = result_first.metric_values();
        let second = result_second.metric_values();
        assert!((first[0] - expected[0]).abs() < 1.0e-4);
        assert!((first[1] - expected[1]).abs() < 1.0e-4);
        assert!((first[2] - expected[2]).abs() < 1.0e-4);
        assert!((second[0] - expected[0]).abs() < 1.0e-4);
        assert!((second[1] - expected[2]).abs() < 1.0e-4);
        assert!((second[2] - expected[1]).abs() < 1.0e-4);
    }

    #[test]
    fn cuda_decision_path_score_rejects_unsupported_metrics_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_decision_path_score() {
            return;
        }

        let rows = 4u64;
        let cols = 1u32;
        let features = vec![0.0, 0.5, 1.0, 1.5];
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let terms = vec![GafimeDecisionPathTerm {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_GT,
            threshold: 0.75,
            ..Default::default()
        }];
        let offsets = vec![0u32, 1];
        let metrics = vec![GAFIME_METRIC_MUTUAL_INFO];
        let mut result = TestResultTable::new(1, 1, 1);
        let err = backend
            .decision_path_score(
                &matrix.handle(),
                &terms,
                &offsets,
                &metrics,
                result.raw_mut(),
            )
            .expect_err("MI must be unsupported for compact decision-path score");
        assert!(matches!(
            err,
            GpuSysError::BackendStatus {
                operation: "gafime_gpu_decision_path_score",
                status: gafime_types::GAFIME_STATUS_UNSUPPORTED_BACKEND,
            }
        ));
    }

    #[test]
    fn cuda_device_topk_returns_only_selected_rows_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 4;
        let cols = 3;
        let features = vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
            return;
        };
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 2,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut result = TestResultTable::new(2, 1, 2);
        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(stats.launched_chunks, 1);
        assert_eq!(stats.rows_written, 2);
        assert_eq!(result.raw.row_count, 2);
        assert_eq!(result.combo_indices(), &[0, 1]);
        assert_eq!(result.ranks(), &[0, 1]);
        assert_eq!(result.candidate_ids(), &[0, 1]);
        let values = result.metric_values();
        assert!((values[0] - 1.0).abs() < 1.0e-5);
        assert!((values[1] - 1.0).abs() < 1.0e-5);
        assert!((values[2] + 1.0).abs() < 1.0e-5);
        assert!((values[3] - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn cuda_device_topk_keeps_large_rank_scratch_bounded_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 4u64;
        let cols = 600u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows {
            features.extend(std::iter::repeat(row as f32).take(cols as usize));
        }
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            (0..cols).collect(),
            vec![GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 400,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut result = TestResultTable::new(400, 1, 1);
        execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(result.raw.row_count, 400);
        assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
        assert!(result.metric_values().iter().all(|value| *value > 0.999));
    }

    #[test]
    fn cuda_graph_flag_replays_same_continuous_result_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let capability = backend.graph_capability().unwrap();
        assert_eq!(capability.graph_mode, GAFIME_GRAPH_STREAM_CAPTURE);
        assert_eq!(capability.supports_device_ranking, 1);

        let rows = 32u64;
        let cols = 6u32;
        let (features, target) = parity_dataset(rows, cols);
        let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
            return;
        };
        matrix.upload(&features, &target).unwrap();

        let config = continuous_config(GAFIME_BACKEND_CUDA);
        let normal_prepared = prepare_continuous_execution(&config, rows, cols).unwrap();
        let graph_plan = prepare_continuous_execution(&config, rows, cols)
            .unwrap()
            .into_plan()
            .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

        let mut normal_result = TestResultTable::new(
            normal_prepared.result_capacity(),
            normal_prepared.result_max_arity(),
            normal_prepared.result_metric_count(),
        );
        execute_plan(
            &mut backend,
            &matrix.handle(),
            normal_prepared.plan(),
            normal_result.raw_mut(),
        )
        .unwrap();

        let mut first_graph_result = TestResultTable::new(
            normal_prepared.result_capacity(),
            normal_prepared.result_max_arity(),
            normal_prepared.result_metric_count(),
        );
        let first_stats = execute_plan(
            &mut backend,
            &matrix.handle(),
            &graph_plan,
            first_graph_result.raw_mut(),
        )
        .unwrap();
        assert_eq!(first_stats.graph_replays, 1);
        assert_ne!(
            first_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0
        );

        let mut second_graph_result = TestResultTable::new(
            normal_prepared.result_capacity(),
            normal_prepared.result_max_arity(),
            normal_prepared.result_metric_count(),
        );
        let second_stats = execute_plan(
            &mut backend,
            &matrix.handle(),
            &graph_plan,
            second_graph_result.raw_mut(),
        )
        .unwrap();
        assert_eq!(second_stats.graph_replays, 1);

        assert_eq!(
            normal_result.raw.row_count,
            first_graph_result.raw.row_count
        );
        assert_eq!(
            normal_result.combo_indices(),
            first_graph_result.combo_indices()
        );
        assert_eq!(
            first_graph_result.combo_indices(),
            second_graph_result.combo_indices()
        );
        for ((normal, first), second) in normal_result
            .metric_values()
            .iter()
            .zip(first_graph_result.metric_values())
            .zip(second_graph_result.metric_values())
        {
            assert!((*normal - *first).abs() <= 5.0e-4);
            assert!((*first - *second).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn cuda_continuous_cached_target_stats_refresh_after_target_update() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        continuous_cached_target_stats_refresh_after_target_update(
            &mut backend,
            GAFIME_BACKEND_CUDA,
        );
    }

    #[test]
    fn cuda_permutation_protocol_preserves_observed_metrics_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 4u32;
        let (features, target) = parity_dataset(rows, cols);
        let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
            return;
        };
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2, 3],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_permutations(GafimePermutationSchedule {
            permutation_count: 4,
            seed: 123,
            ..Default::default()
        })
        .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

        let mut result = TestResultTable::new(4, 1, 2);
        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(stats.launched_chunks, 1);
        assert_eq!(stats.graph_replays, 1);
        assert_eq!(stats.rows_written, 4);
        assert_ne!(result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED, 0);
        let observed_metrics = result.metric_values().to_vec();

        let no_permutation_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2, 3],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );
        let mut restored_result = TestResultTable::new(4, 1, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &no_permutation_plan,
            restored_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(result.combo_indices(), restored_result.combo_indices());
        for (left, right) in observed_metrics
            .iter()
            .zip(restored_result.metric_values().iter())
        {
            assert!((*left - *right).abs() <= 5.0e-4);
        }
    }

    #[test]
    fn cuda_reports_permutation_pvalues_when_library_exposes_optional_abi() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_permutation_pvalues() {
            return;
        }

        let rows = 64u64;
        let cols = 2u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let signal = row as f32;
            let noise = ((row * 17) % 29) as f32;
            features.extend([signal, noise]);
            target.push(signal);
        }

        let Ok(matrix) = backend.alloc_matrix(rows, cols) else {
            return;
        };
        matrix.upload(&features, &target).unwrap();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_permutations(GafimePermutationSchedule {
            permutation_count: 16,
            seed: 99,
            ..Default::default()
        })
        .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);

        let mut result = TestResultTable::new(2, 1, 2);
        execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();
        let pvalues = backend
            .permutation_pvalues(
                &matrix.handle(),
                plan.protocol(),
                result.candidate_ids(),
                result.metric_values(),
                2,
            )
            .unwrap()
            .expect("CUDA payload should expose permutation p-value ABI");

        assert_eq!(pvalues.len(), 4);
        assert!(pvalues.iter().all(|value| value.is_finite()));
        assert!(pvalues.iter().all(|&value| value > 0.0 && value <= 1.0));
        assert!(pvalues[0] <= 0.25, "signal pearson p-value={}", pvalues[0]);
        assert!(pvalues[1] <= 0.25, "signal r2 p-value={}", pvalues[1]);
    }

    #[test]
    fn cuda_permutation_maxt_includes_hidden_family_candidates_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        if !backend.supports_permutation_pvalues() {
            return;
        }

        let rows = 64u64;
        let cols = 2u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            features.extend([1.0, row as f32]);
            target.push(((row * 17 + 3) % 61) as f32);
        }
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let permutations = GafimePermutationSchedule {
            permutation_count: 32,
            seed: 0x5A17,
            ..Default::default()
        };
        let selected_only = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0],
            vec![GAFIME_METRIC_PEARSON],
        )
        .with_permutations(permutations);
        let full_family = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON],
        )
        .with_permutations(permutations);

        let selected_p = backend
            .permutation_pvalues(&matrix.handle(), selected_only.protocol(), &[0], &[0.1], 1)
            .unwrap()
            .unwrap()[0];
        let family_p = backend
            .permutation_pvalues(&matrix.handle(), full_family.protocol(), &[0], &[0.1], 1)
            .unwrap()
            .unwrap()[0];

        let floor = 1.0 / (permutations.permutation_count as f32 + 1.0);
        assert!((selected_p - floor).abs() <= f32::EPSILON);
        assert!(
            family_p > selected_p,
            "hidden family candidate must raise maxT p-value: selected={selected_p}, family={family_p}"
        );
    }

    #[test]
    fn cuda_mutual_info_metric_returns_finite_signal_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 128u64;
        let cols = 2u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let x0 = (row % 16) as f32 / 15.0;
            let x1 = ((row * 7) % 23) as f32 / 22.0;
            features.extend([x0, x1]);
            target.push(if x0 > 0.5 { 1.0 } else { 0.0 });
        }
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_MUTUAL_INFO],
        );
        let mut result = TestResultTable::new(2, 1, 1);
        execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(result.raw.row_count, 2);
        let values = result.metric_values();
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[0] >= 0.0);
        assert!(values[0] > values[1]);
    }

    #[test]
    fn cuda_fixed_mi_extreme_bin_mapping_matches_cpu_when_available() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 512u64;
        let cols = 2u32;
        let mut wide = Vec::with_capacity(rows as usize);
        let mut subnormal = Vec::with_capacity(rows as usize);
        let mut target = Vec::with_capacity(rows as usize);
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows as usize {
            let wide_value = if row % 2 == 0 { -f32::MAX } else { f32::MAX };
            let subnormal_value = f32::from_bits((row % 9) as u32);
            let target_value = (row % 9) as f32;
            wide.push(wide_value);
            subnormal.push(subnormal_value);
            target.push(target_value);
            features.extend([wide_value, subnormal_value]);
        }

        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_MUTUAL_INFO],
        );
        let mut result = TestResultTable::new(2, 1, 1);
        execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        let expected_wide = gafime_cpu::kernels::mutual_info_fixed(&wide, &target, 8);
        let expected_subnormal = gafime_cpu::kernels::mutual_info_fixed(&subnormal, &target, 8);
        let actual = result.metric_values();
        assert_eq!(expected_wide, 0.0);
        assert_eq!(actual[0].to_bits(), expected_wide.to_bits());
        assert!(
            (actual[1] - expected_subnormal).abs() <= 1.0e-5,
            "subnormal MI mismatch: CUDA={}, CPU={expected_subnormal}",
            actual[1]
        );
    }

    #[test]
    fn cuda_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available() {
        let _cuda_guard = cuda_test_lock();
        let Some(mut cuda_backend) = cuda_backend_for_specialization_test() else {
            return;
        };
        assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
            &mut cuda_backend,
            GAFIME_BACKEND_CUDA,
            MI_TEMPLATE_BIN_LEVELS,
        );
    }

    #[test]
    fn cuda_matches_cpu_for_configured_continuous_plan_arity_1_to_5() {
        let _cuda_guard = cuda_test_lock();
        let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 6u32;
        let (features, target) = parity_dataset(rows, cols);

        let mut cpu_config = continuous_config(GAFIME_BACKEND_CPU);
        let cpu_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
        let mut cpu_result = TestResultTable::new(
            cpu_prepared.result_capacity(),
            cpu_prepared.result_max_arity(),
            cpu_prepared.result_metric_count(),
        );
        let cpu_stats = execute_plan(
            &mut cpu_backend,
            &cpu_matrix.handle(),
            cpu_prepared.plan(),
            cpu_result.raw_mut(),
        )
        .unwrap();

        let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
        cuda_matrix.upload(&features, &target).unwrap();
        let cuda_config = continuous_config(GAFIME_BACKEND_CUDA);
        let cuda_prepared = prepare_continuous_execution(&cuda_config, rows, cols).unwrap();
        let mut cuda_result = TestResultTable::new(
            cuda_prepared.result_capacity(),
            cuda_prepared.result_max_arity(),
            cuda_prepared.result_metric_count(),
        );
        let cuda_stats = execute_plan(
            &mut cuda_backend,
            &cuda_matrix.handle(),
            cuda_prepared.plan(),
            cuda_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(cpu_prepared.result_capacity(), 62);
        assert_eq!(cuda_prepared.result_capacity(), 62);
        assert_eq!(cpu_stats.launched_chunks, 5);
        assert_eq!(cuda_stats.launched_chunks, 5);
        assert_eq!(cpu_result.raw.row_count, cuda_result.raw.row_count);
        assert_eq!(cpu_result.combo_indices(), cuda_result.combo_indices());

        for (index, (&cpu_value, &cuda_value)) in cpu_result
            .metric_values()
            .iter()
            .zip(cuda_result.metric_values())
            .enumerate()
        {
            let delta = (cpu_value - cuda_value).abs();
            assert!(
                delta <= 5.0e-4,
                "metric mismatch at {index}: cpu={cpu_value} cuda={cuda_value} delta={delta}"
            );
        }

        cpu_config.backend_kind = GAFIME_BACKEND_CUDA;
        let explicit_cuda_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
        assert_eq!(
            explicit_cuda_prepared.plan().protocol().backend_kind,
            GAFIME_BACKEND_CUDA
        );
    }

    #[test]
    fn rocm_matches_cpu_for_continuous_pearson_r2_when_library_is_available() {
        // ROCm supports the continuous pearson/r2 subset on gfx1150; validate it
        // against the CPU reference over the same plan (skips without the payload).
        let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 6u32;
        let (features, target) = parity_dataset(rows, cols);

        let cpu_config = continuous_config(GAFIME_BACKEND_CPU);
        let cpu_prepared = prepare_continuous_execution(&cpu_config, rows, cols).unwrap();
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
        let mut cpu_result = TestResultTable::new(
            cpu_prepared.result_capacity(),
            cpu_prepared.result_max_arity(),
            cpu_prepared.result_metric_count(),
        );
        execute_plan(
            &mut cpu_backend,
            &cpu_matrix.handle(),
            cpu_prepared.plan(),
            cpu_result.raw_mut(),
        )
        .unwrap();

        let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
        rocm_matrix.upload(&features, &target).unwrap();
        let rocm_config = continuous_config(GAFIME_BACKEND_ROCM);
        let rocm_prepared = prepare_continuous_execution(&rocm_config, rows, cols).unwrap();
        let mut rocm_result = TestResultTable::new(
            rocm_prepared.result_capacity(),
            rocm_prepared.result_max_arity(),
            rocm_prepared.result_metric_count(),
        );
        let rocm_stats = execute_plan(
            &mut rocm_backend,
            &rocm_matrix.handle(),
            rocm_prepared.plan(),
            rocm_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(
            rocm_prepared.plan().protocol().backend_kind,
            GAFIME_BACKEND_ROCM
        );
        assert_eq!(rocm_backend.backend_kind(), GAFIME_BACKEND_ROCM);
        assert_eq!(rocm_stats.rows_written, cpu_result.raw.row_count);
        assert_eq!(cpu_result.raw.row_count, rocm_result.raw.row_count);
        assert_eq!(cpu_result.combo_indices(), rocm_result.combo_indices());
        for (index, (&cpu_value, &rocm_value)) in cpu_result
            .metric_values()
            .iter()
            .zip(rocm_result.metric_values())
            .enumerate()
        {
            let delta = (cpu_value - rocm_value).abs();
            assert!(
                delta <= 5.0e-4,
                "metric mismatch at {index}: cpu={cpu_value} rocm={rocm_value} delta={delta}"
            );
        }
    }

    #[test]
    fn rocm_mutual_info_detects_signal_and_matches_cuda_when_available() {
        let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

        let rows = 128u64;
        let cols = 2u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let x0 = (row % 16) as f32 / 15.0;
            let x1 = ((row * 7) % 23) as f32 / 22.0;
            features.extend([x0, x1]);
            target.push(if x0 > 0.5 { 1.0 } else { 0.0 });
        }

        let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
        rocm_matrix.upload(&features, &target).unwrap();
        let rocm_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_ROCM,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_MUTUAL_INFO],
        );
        let mut rocm_result = TestResultTable::new(2, 1, 1);
        execute_plan(
            &mut rocm_backend,
            &rocm_matrix.handle(),
            &rocm_plan,
            rocm_result.raw_mut(),
        )
        .unwrap();
        let rocm_mi = rocm_result.metric_values().to_vec();
        assert!(rocm_mi[0].is_finite() && rocm_mi[1].is_finite());
        assert!(rocm_mi[0] >= 0.0);
        assert!(
            rocm_mi[0] > rocm_mi[1],
            "MI must detect the x0->target signal: {rocm_mi:?}"
        );

        // The ROCm MI kernel is a verbatim port of the CUDA fixed-binning kernel,
        // so their outputs match within fp tolerance on the same input.
        let _cuda_guard = cuda_test_lock();
        if let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) {
            let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
            cuda_matrix.upload(&features, &target).unwrap();
            let cuda_plan = CompiledPlan::single_chunk(
                GAFIME_BACKEND_CUDA,
                rows,
                cols,
                GAFIME_FAMILY_CONTINUOUS,
                1,
                vec![0, 1],
                vec![GAFIME_METRIC_MUTUAL_INFO],
            );
            let mut cuda_result = TestResultTable::new(2, 1, 1);
            execute_plan(
                &mut cuda_backend,
                &cuda_matrix.handle(),
                &cuda_plan,
                cuda_result.raw_mut(),
            )
            .unwrap();
            for (i, (&r, &c)) in rocm_mi.iter().zip(cuda_result.metric_values()).enumerate() {
                assert!(
                    (r - c).abs() <= 1.0e-3,
                    "ROCm/CUDA MI mismatch at {i}: rocm={r} cuda={c}"
                );
            }
        }
    }

    #[test]
    fn rocm_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available() {
        let Some(mut rocm_backend) = rocm_backend_for_specialization_test() else {
            return;
        };
        assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
            &mut rocm_backend,
            GAFIME_BACKEND_ROCM,
            MI_TEMPLATE_BIN_LEVELS,
        );
    }

    #[test]
    fn rocm_adaptive_mi_96_matches_cpu_for_arity_1_to_5_when_library_is_available() {
        let require_wave64 = env::var_os("GAFIME_REQUIRE_ROCM_WAVE64_MI").is_some();
        if !require_wave64 {
            return;
        }
        let mut rocm_backend = GpuBackend::rocm_from_env(0)
            .unwrap_or_else(|error| panic!("required wave64 ROCm payload failed to load: {error}"));
        let device_info = rocm_backend.device_info().unwrap();
        assert_eq!(device_info.warp_size, 64, "wave64 MI validation required");
        assert_ne!(
            device_info.flags & GAFIME_GPU_DEVICE_FLAG_AMD_CDNA,
            0,
            "wave64 MI validation requires a CDNA device"
        );
        assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
            &mut rocm_backend,
            GAFIME_BACKEND_ROCM,
            &[96],
        );
    }

    #[test]
    fn rocm_device_topk_selects_by_primary_metric_when_available() {
        // ROCm device top-k should keep the same deterministic primary-metric
        // ordering as the CPU/CUDA paths.
        let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };
        if std::env::var_os("GAFIME_REQUIRE_CURRENT_DEVICE_RANKING").is_some() {
            assert_eq!(
                backend.graph_capability().unwrap().supports_device_ranking,
                1
            );
        }

        let rows = 4;
        let cols = 3;
        let features = vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_ROCM,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 2,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut result = TestResultTable::new(2, 1, 2);
        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 2);
        assert_eq!(result.raw.row_count, 2);
        assert_eq!(result.combo_indices(), &[0, 1]);
        assert_eq!(result.ranks(), &[0, 1]);
        assert_eq!(result.candidate_ids(), &[0, 1]);
        let values = result.metric_values();
        assert!((values[1] - 1.0).abs() < 1.0e-5); // r2 of feature 0
        assert!((values[3] - 1.0).abs() < 1.0e-5); // r2 of feature 1
    }

    #[test]
    fn rocm_device_topk_keeps_large_rank_scratch_bounded_when_library_is_available() {
        let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

        let rows = 4u64;
        let cols = 600u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows {
            features.extend(std::iter::repeat(row as f32).take(cols as usize));
        }
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_ROCM,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            (0..cols).collect(),
            vec![GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 400,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut result = TestResultTable::new(400, 1, 1);
        execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

        assert_eq!(result.raw.row_count, 400);
        assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
        assert!(result.metric_values().iter().all(|value| *value > 0.999));
    }

    #[test]
    fn metal_descriptor_cache_generation_refreshes_reused_addresses_when_available() {
        let _metal_guard = metal_test_lock();
        let Some(mut backend) = metal_backend_for_test() else {
            return;
        };
        assert!(
            backend.supports_descriptor_generation(),
            "configured Metal payload must advertise descriptor-generation support"
        );

        let rows = 4u64;
        let cols = 2u32;
        let features = vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_METAL,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0],
            vec![GAFIME_METRIC_PEARSON],
        );
        let mut descriptors = vec![0u32];
        let descriptor_address = descriptors.as_ptr();
        let mut first_protocol = *plan.protocol();
        first_protocol.combo_indices.ptr = descriptor_address;
        first_protocol.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        first_protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 101;

        let mut first_result = TestResultTable::new(1, 1, 1);
        backend
            .execute(matrix.handle(), &first_protocol, first_result.raw_mut())
            .unwrap();
        assert_eq!(first_result.combo_indices(), &[0]);
        assert!(first_result.metric_values()[0] > 0.999);

        descriptors[0] = 1;
        assert_eq!(descriptors.as_ptr(), descriptor_address);
        let mut second_protocol = first_protocol;
        second_protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 102;
        let mut second_result = TestResultTable::new(1, 1, 1);
        backend
            .execute(matrix.handle(), &second_protocol, second_result.raw_mut())
            .unwrap();
        assert_eq!(second_result.combo_indices(), &[1]);
        assert!(second_result.metric_values()[0] < -0.999);

        descriptors[0] = 0;
        assert_eq!(descriptors.as_ptr(), descriptor_address);
        let mut replay_result = TestResultTable::new(1, 1, 1);
        backend
            .execute(matrix.handle(), &second_protocol, replay_result.raw_mut())
            .unwrap();
        assert!(replay_result.metric_values()[0] < -0.999);
        assert!(
            (replay_result.metric_values()[0] - second_result.metric_values()[0]).abs() < 1.0e-5
        );
    }

    #[test]
    fn metal_device_topk_covers_split_directions_ties_and_large_k_when_available() {
        let _metal_guard = metal_test_lock();
        let Some(mut backend) = metal_backend_for_test() else {
            return;
        };
        if std::env::var_os("GAFIME_REQUIRE_CURRENT_DEVICE_RANKING").is_some() {
            assert_eq!(
                backend.graph_capability().unwrap().supports_device_ranking,
                1
            );
        }

        {
            let rows = 5u64;
            let cols = 4u32;
            let mut features = Vec::with_capacity(rows as usize * cols as usize);
            let mut target = Vec::with_capacity(rows as usize);
            for row in 0..rows as usize {
                let value = row as f32;
                features.extend([value, value * value, (row % 2) as f32, 1.0]);
                target.push(value);
            }
            let matrix = backend.alloc_matrix(rows, cols).unwrap();
            matrix.upload(&features, &target).unwrap();

            for (descending, expected) in [(1, [0u32, 1]), (0, [2u32, 3])] {
                let plan = CompiledPlan::single_chunk(
                    GAFIME_BACKEND_METAL,
                    rows,
                    cols,
                    GAFIME_FAMILY_CONTINUOUS,
                    1,
                    (0..cols).collect(),
                    vec![GAFIME_METRIC_R2],
                )
                .with_rank(GafimeRankSpec {
                    top_k: 2,
                    primary_metric: GAFIME_METRIC_R2,
                    descending,
                    include_ties: 0,
                    reserved: [0; 4],
                });
                let mut result = TestResultTable::new(2, 1, 1);
                execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

                assert_eq!(result.combo_indices(), &expected);
                assert_eq!(result.candidate_ids(), &expected.map(u64::from));
                if descending != 0 {
                    assert!(result.metric_values()[0] > 0.999);
                    assert!(result.metric_values()[1] > 0.8);
                } else {
                    assert!(result
                        .metric_values()
                        .iter()
                        .all(|value| value.abs() < 1.0e-5));
                }
            }
        }

        let rows = 4u64;
        let cols = 600u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows {
            features.extend(std::iter::repeat(row as f32).take(cols as usize));
        }
        let target = vec![0.0, 1.0, 2.0, 3.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        for descending in [0, 1] {
            let plan = CompiledPlan::single_chunk(
                GAFIME_BACKEND_METAL,
                rows,
                cols,
                GAFIME_FAMILY_CONTINUOUS,
                1,
                (0..cols).collect(),
                vec![GAFIME_METRIC_R2],
            )
            .with_rank(GafimeRankSpec {
                top_k: 400,
                primary_metric: GAFIME_METRIC_R2,
                descending,
                include_ties: 0,
                reserved: [0; 4],
            });
            let mut result = TestResultTable::new(400, 1, 1);
            execute_plan(&mut backend, &matrix.handle(), &plan, result.raw_mut()).unwrap();

            assert_eq!(result.raw.row_count, 400);
            assert_eq!(result.combo_indices(), (0..400).collect::<Vec<_>>());
            assert_eq!(result.candidate_ids(), (0u64..400).collect::<Vec<_>>());
            assert!(result.metric_values().iter().all(|value| *value > 0.999));
        }

        let oversized_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_METAL,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            (0..cols).collect(),
            vec![GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 700,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut oversized_result = TestResultTable::new(700, 1, 1);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &oversized_plan,
            oversized_result.raw_mut(),
        )
        .unwrap();
        assert_eq!(oversized_result.raw.row_count, u64::from(cols));
        assert_eq!(
            oversized_result.combo_indices(),
            (0..cols).collect::<Vec<_>>()
        );
    }

    #[test]
    fn metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available() {
        const DEFAULT_METAL_PARITY_TOLERANCE: f32 = 2.0e-3;

        let _metal_guard = metal_test_lock();
        let Some(mut metal_backend) = metal_backend_for_test() else {
            return;
        };
        let tolerance = match env::var("GAFIME_METAL_PARITY_TOLERANCE") {
            Ok(value) => value
                .parse::<f32>()
                .expect("GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"),
            Err(env::VarError::NotPresent) => DEFAULT_METAL_PARITY_TOLERANCE,
            Err(env::VarError::NotUnicode(_)) => {
                panic!("GAFIME_METAL_PARITY_TOLERANCE must be valid UTF-8")
            }
        };
        assert!(
            tolerance.is_finite() && tolerance > 0.0,
            "GAFIME_METAL_PARITY_TOLERANCE must be a finite positive float"
        );

        let rows = 160u64;
        let cols = 5u32;
        for inject_nonfinite in [false, true] {
            let (features, target) = metal_parity_dataset(rows, cols, inject_nonfinite);
            let prepare = |backend_kind| {
                let mut config = EngineConfig::default();
                config.backend_kind = backend_kind;
                config.metric_ids = vec![
                    GAFIME_METRIC_PEARSON,
                    GAFIME_METRIC_R2,
                    GAFIME_METRIC_MUTUAL_INFO,
                    GAFIME_METRIC_SPEARMAN,
                ];
                config.mi_bins = 96;
                config.mi_approximate = true;
                config.permutation_tests = 0;
                config.budget.max_comb_size = 5;
                config.budget.max_combinations_per_k = 100;
                prepare_continuous_execution(&config, rows, cols).unwrap()
            };

            let cpu_prepared = prepare(GAFIME_BACKEND_CPU);
            let cpu_matrix =
                CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
            let mut cpu_backend = CpuBackend;
            let mut cpu_result = TestResultTable::new(
                cpu_prepared.result_capacity(),
                cpu_prepared.result_max_arity(),
                cpu_prepared.result_metric_count(),
            );
            execute_plan(
                &mut cpu_backend,
                &cpu_matrix.handle(),
                cpu_prepared.plan(),
                cpu_result.raw_mut(),
            )
            .unwrap();

            let metal_prepared = prepare(GAFIME_BACKEND_METAL);
            let metal_matrix = metal_backend.alloc_matrix(rows, cols).unwrap();
            metal_matrix.upload(&features, &target).unwrap();
            let mut metal_result = TestResultTable::new(
                metal_prepared.result_capacity(),
                metal_prepared.result_max_arity(),
                metal_prepared.result_metric_count(),
            );
            execute_plan(
                &mut metal_backend,
                &metal_matrix.handle(),
                metal_prepared.plan(),
                metal_result.raw_mut(),
            )
            .unwrap();

            assert_eq!(cpu_result.raw.row_count, 31);
            assert_eq!(cpu_result.raw.row_count, metal_result.raw.row_count);
            assert_eq!(cpu_result.combo_indices(), metal_result.combo_indices());
            assert_eq!(cpu_result.candidate_ids(), metal_result.candidate_ids());
            for (index, (&cpu_value, &metal_value)) in cpu_result
                .metric_values()
                .iter()
                .zip(metal_result.metric_values())
                .enumerate()
            {
                let delta = (cpu_value - metal_value).abs();
                assert!(
                    cpu_value.is_finite() && metal_value.is_finite() && delta <= tolerance,
                    "Metal parity mismatch at metric value {index} (nonfinite={inject_nonfinite}): \
                     cpu={cpu_value} metal={metal_value} delta={delta} tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn cuda_spearman_matches_cpu_when_library_is_available() {
        // Spearman = pearson on ranks; the CUDA count-based ranks must match the
        // CPU rankdata (including average-tie ranks) within fp tolerance.
        let _cuda_guard = cuda_test_lock();
        let Ok(mut cuda_backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 48u64;
        let cols = 3u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for r in 0..rows as usize {
            let a = r as f32 * 0.13; // strictly increasing
            let b = ((r * 7) % 17) as f32; // repeated values -> ties
            let c = (rows as usize - r) as f32; // strictly decreasing
            features.extend([a, b, c]);
            target.push(a * a * a); // strictly monotone in feature 0
        }

        let cpu_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_SPEARMAN],
        );
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
        let mut cpu_result = TestResultTable::new(3, 1, 1);
        execute_plan(
            &mut cpu_backend,
            &cpu_matrix.handle(),
            &cpu_plan,
            cpu_result.raw_mut(),
        )
        .unwrap();

        let cuda_matrix = cuda_backend.alloc_matrix(rows, cols).unwrap();
        cuda_matrix.upload(&features, &target).unwrap();
        let cuda_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CUDA,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_SPEARMAN],
        );
        let mut cuda_result = TestResultTable::new(3, 1, 1);
        execute_plan(
            &mut cuda_backend,
            &cuda_matrix.handle(),
            &cuda_plan,
            cuda_result.raw_mut(),
        )
        .unwrap();

        let cpu_vals = cpu_result.metric_values();
        let cuda_vals = cuda_result.metric_values();
        // feature 0 is strictly monotone in target -> spearman == 1; feature 2 is
        // strictly anti-monotone -> spearman == -1.
        assert!(cpu_vals[0] > 0.999, "cpu spearman(f0)={}", cpu_vals[0]);
        assert!(cpu_vals[2] < -0.999, "cpu spearman(f2)={}", cpu_vals[2]);
        for (i, (&c, &g)) in cpu_vals.iter().zip(cuda_vals).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-4,
                "spearman mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    #[test]
    fn rocm_spearman_matches_cpu_when_library_is_available() {
        let Ok(mut rocm_backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

        let rows = 48u64;
        let cols = 3u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for r in 0..rows as usize {
            let a = r as f32 * 0.13;
            let b = ((r * 7) % 17) as f32;
            let c = (rows as usize - r) as f32;
            features.extend([a, b, c]);
            target.push(a * a * a);
        }

        let cpu_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_SPEARMAN],
        );
        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
        let mut cpu_result = TestResultTable::new(3, 1, 1);
        execute_plan(
            &mut cpu_backend,
            &cpu_matrix.handle(),
            &cpu_plan,
            cpu_result.raw_mut(),
        )
        .unwrap();

        let rocm_matrix = rocm_backend.alloc_matrix(rows, cols).unwrap();
        rocm_matrix.upload(&features, &target).unwrap();
        let rocm_plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_ROCM,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_SPEARMAN],
        );
        let mut rocm_result = TestResultTable::new(3, 1, 1);
        execute_plan(
            &mut rocm_backend,
            &rocm_matrix.handle(),
            &rocm_plan,
            rocm_result.raw_mut(),
        )
        .unwrap();

        let cpu_vals = cpu_result.metric_values();
        let rocm_vals = rocm_result.metric_values();
        assert!(cpu_vals[0] > 0.999);
        assert!(cpu_vals[2] < -0.999);
        for (i, (&c, &g)) in cpu_vals.iter().zip(rocm_vals).enumerate() {
            assert!(
                (c - g).abs() <= 1.0e-4,
                "spearman mismatch at {i}: cpu={c} rocm={g}"
            );
        }
    }

    #[test]
    fn cuda_graph_captures_whole_multi_arity_sweep_when_available() {
        // The CUDA host captures the entire multi-arity sweep (every chunk +
        // metric) into ONE graph, not one graph per shape. Validate that a
        // multi-chunk plan replays as a single graph with results identical to a
        // normal launch.
        let _cuda_guard = cuda_test_lock();
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 5u32;
        let (features, target) = parity_dataset(rows, cols);
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let request = |flags: u32| {
            let mut plan = build_continuous_plan(ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CUDA,
                n_samples: rows,
                n_features: cols,
                max_arity: 3,
                max_combinations_per_arity: 1_000,
                metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
                mi_bins: 96,
                rank: Default::default(),
            })
            .unwrap();
            if flags != 0 {
                plan = plan.with_flags(flags);
            }
            plan
        };

        let graph_plan = request(GAFIME_LAUNCH_FLAG_GRAPH);
        assert!(
            graph_plan.chunks().len() >= 3,
            "arity 1..3 should produce several chunks (a real sweep)"
        );
        let planned: u64 = graph_plan.chunks().iter().map(|c| c.combo_count).sum();

        let mut graph_result = TestResultTable::new(planned, 3, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &graph_plan,
            graph_result.raw_mut(),
        )
        .unwrap();
        assert_ne!(
            graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0,
            "the whole multi-arity sweep must replay as one graph"
        );

        let normal_plan = request(0);
        let mut normal_result = TestResultTable::new(planned, 3, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &normal_plan,
            normal_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(graph_result.raw.row_count, normal_result.raw.row_count);
        assert_eq!(graph_result.combo_indices(), normal_result.combo_indices());
        for (g, n) in graph_result
            .metric_values()
            .iter()
            .zip(normal_result.metric_values())
        {
            assert!((g - n).abs() <= 5.0e-4, "graph vs normal: {g} vs {n}");
        }
    }

    #[test]
    fn rocm_graph_captures_and_replays_the_sweep_when_available() {
        // ROCm device-copy stream-capture: the multi-arity sweep is captured once
        // and replayed; results must match a normal launch, and a second run must
        // reuse the cached graph.
        let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 4u32;
        let (features, target) = parity_dataset(rows, cols);
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target).unwrap();

        let request = |flags: u32| {
            let mut plan = build_continuous_plan(ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_ROCM,
                n_samples: rows,
                n_features: cols,
                max_arity: 2,
                max_combinations_per_arity: 1_000,
                metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
                mi_bins: 96,
                rank: Default::default(),
            })
            .unwrap();
            if flags != 0 {
                plan = plan.with_flags(flags);
            }
            plan
        };

        let graph_plan = request(GAFIME_LAUNCH_FLAG_GRAPH);
        let planned: u64 = graph_plan.chunks().iter().map(|c| c.combo_count).sum();

        let mut graph_result = TestResultTable::new(planned, 2, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &graph_plan,
            graph_result.raw_mut(),
        )
        .unwrap();
        assert_ne!(
            graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0,
            "ROCm should capture + replay the sweep as a graph"
        );

        // Second run reuses the cached graph (same shape/signature).
        let mut graph_result2 = TestResultTable::new(planned, 2, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &request(GAFIME_LAUNCH_FLAG_GRAPH),
            graph_result2.raw_mut(),
        )
        .unwrap();
        assert_ne!(
            graph_result2.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0
        );

        let normal_plan = request(0);
        let mut normal_result = TestResultTable::new(planned, 2, 2);
        execute_plan(
            &mut backend,
            &matrix.handle(),
            &normal_plan,
            normal_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(graph_result.combo_indices(), normal_result.combo_indices());
        for (g, n) in graph_result
            .metric_values()
            .iter()
            .zip(normal_result.metric_values())
        {
            assert!((g - n).abs() <= 5.0e-4, "graph vs normal: {g} vs {n}");
        }
        for (g, n) in graph_result2
            .metric_values()
            .iter()
            .zip(normal_result.metric_values())
        {
            assert!((g - n).abs() <= 5.0e-4);
        }
    }

    #[test]
    fn rocm_continuous_cached_target_stats_refresh_after_target_update() {
        let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };
        continuous_cached_target_stats_refresh_after_target_update(
            &mut backend,
            GAFIME_BACKEND_ROCM,
        );
    }

    fn assert_adaptive_mi_templates_match_cpu_for_arity_1_to_5(
        gpu_backend: &mut GpuBackend,
        backend_kind: u32,
        bins_to_test: &[u32],
    ) {
        assert!(!bins_to_test.is_empty());
        let rows = 73_728u64;
        let cols = 5u32;
        let (features, target) = parity_dataset(rows, cols);
        let prepare = |planned_backend_kind, bins| {
            let mut config = EngineConfig::default();
            config.backend_kind = planned_backend_kind;
            config.metric_ids = vec![GAFIME_METRIC_MUTUAL_INFO];
            config.mi_bins = bins;
            config.mi_approximate = true;
            config.permutation_tests = 0;
            config.budget.max_comb_size = 5;
            config.budget.max_combinations_per_k = 100;
            prepare_continuous_execution(&config, rows, cols).unwrap()
        };

        let cpu_matrix =
            CpuMatrix::from_row_major(rows, cols, features.clone(), target.clone()).unwrap();
        let mut cpu_backend = CpuBackend;
        let gpu_matrix = gpu_backend.alloc_matrix(rows, cols).unwrap();
        gpu_matrix.upload(&features, &target).unwrap();

        for &bins in bins_to_test {
            let cpu_prepared = prepare(GAFIME_BACKEND_CPU, bins);
            let gpu_prepared = prepare(backend_kind, bins);

            let mut cpu_result = TestResultTable::new(
                cpu_prepared.result_capacity(),
                cpu_prepared.result_max_arity(),
                cpu_prepared.result_metric_count(),
            );
            execute_plan(
                &mut cpu_backend,
                &cpu_matrix.handle(),
                cpu_prepared.plan(),
                cpu_result.raw_mut(),
            )
            .unwrap();

            let mut gpu_result = TestResultTable::new(
                gpu_prepared.result_capacity(),
                gpu_prepared.result_max_arity(),
                gpu_prepared.result_metric_count(),
            );
            execute_plan(
                gpu_backend,
                &gpu_matrix.handle(),
                gpu_prepared.plan(),
                gpu_result.raw_mut(),
            )
            .unwrap();

            assert_eq!(cpu_result.raw.row_count, 31);
            assert_eq!(cpu_result.raw.row_count, gpu_result.raw.row_count);
            assert_eq!(cpu_result.combo_indices(), gpu_result.combo_indices());
            assert_eq!(cpu_result.candidate_ids(), gpu_result.candidate_ids());
            for (index, (&cpu_value, &gpu_value)) in cpu_result
                .metric_values()
                .iter()
                .zip(gpu_result.metric_values())
                .enumerate()
            {
                let delta = (cpu_value - gpu_value).abs();
                assert!(
                    cpu_value.is_finite() && gpu_value.is_finite() && delta <= 1.0e-3,
                    "MI mismatch at {index}: backend={backend_kind} bins={bins} \
                     cpu={cpu_value} gpu={gpu_value} delta={delta}"
                );
            }
        }
    }

    fn continuous_config(backend_kind: u32) -> EngineConfig {
        let mut config = EngineConfig::default();
        config.backend_kind = backend_kind;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.permutation_tests = 0;
        config.budget.max_comb_size = 5;
        config.budget.max_combinations_per_k = 10_000;
        config
    }

    fn continuous_cached_target_stats_refresh_after_target_update(
        backend: &mut GpuBackend,
        backend_kind: u32,
    ) {
        let rows = 8u64;
        let cols = 2u32;
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows as usize {
            features.push(row as f32);
            features.push((rows as usize - 1 - row) as f32);
        }
        let target_a = (0..rows as usize).map(|row| row as f32).collect::<Vec<_>>();
        let target_b = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
        matrix.upload(&features, &target_a).unwrap();

        let graph_plan = CompiledPlan::single_chunk(
            backend_kind,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);
        let normal_plan = CompiledPlan::single_chunk(
            backend_kind,
            rows,
            cols,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );

        let mut first_graph_result = TestResultTable::new(2, 1, 2);
        execute_plan(
            backend,
            &matrix.handle(),
            &graph_plan,
            first_graph_result.raw_mut(),
        )
        .unwrap();
        assert_ne!(
            first_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0
        );

        matrix.update_target(&target_b).unwrap();

        let mut updated_graph_result = TestResultTable::new(2, 1, 2);
        execute_plan(
            backend,
            &matrix.handle(),
            &graph_plan,
            updated_graph_result.raw_mut(),
        )
        .unwrap();
        assert_ne!(
            updated_graph_result.raw.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
            0
        );

        let mut updated_normal_result = TestResultTable::new(2, 1, 2);
        execute_plan(
            backend,
            &matrix.handle(),
            &normal_plan,
            updated_normal_result.raw_mut(),
        )
        .unwrap();

        assert_eq!(
            updated_graph_result.combo_indices(),
            updated_normal_result.combo_indices()
        );
        for (idx, (&graph, &normal)) in updated_graph_result
            .metric_values()
            .iter()
            .zip(updated_normal_result.metric_values())
            .enumerate()
        {
            assert!(
                (graph - normal).abs() <= 1.0e-5,
                "metric {idx}: graph={graph} normal={normal}"
            );
        }
        assert!(
            (first_graph_result.metric_values()[0] - updated_graph_result.metric_values()[0]).abs()
                > 1.0e-3,
            "target update must materially change the cached-target fast path"
        );
    }

    fn parity_dataset(rows: u64, cols: u32) -> (Vec<f32>, Vec<f32>) {
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        for row in 0..rows as usize {
            let r = row as f32 + 1.0;
            for col in 0..cols as usize {
                let c = col as f32 + 1.0;
                let wave = ((row * (col + 3)) % 11) as f32 * 0.017;
                features.push((r * 0.031 * c) + wave + (c * c * 0.003));
            }
        }
        let target = (0..rows as usize)
            .map(|row| {
                let r = row as f32 + 1.0;
                (r * 0.071) + ((row % 5) as f32 * 0.043) - ((row % 3) as f32 * 0.019)
            })
            .collect();
        (features, target)
    }

    fn metal_parity_dataset(rows: u64, cols: u32, inject_nonfinite: bool) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(cols, 5);
        let mut features = Vec::with_capacity(rows as usize * cols as usize);
        let mut target = Vec::with_capacity(rows as usize);
        for row in 0..rows as usize {
            let x0 = 1_000_000.0 + ((row * 17) % 127) as f32 * 0.25;
            let x1 = -500_000.0 + ((row * 29) % 113) as f32 * 0.125;
            let x2 = ((row * 7) % 41) as f32 - 20.0;
            let x3 = ((row * 13) % 67) as f32 * 0.5 + (row % 3) as f32 * 0.125;
            let x4 = ((row * 31) % 59) as f32 * 0.75 - 20.0;
            features.extend([x0, x1, x2, x3, x4]);
            target.push(
                250_000.0 + (x0 - 1_000_000.0) * 0.375 - (x1 + 500_000.0) * 0.25
                    + x2 * 1.125
                    + x4 * 0.5
                    + ((row * 19) % 23) as f32 * 0.0625,
            );
        }
        if inject_nonfinite {
            features[17 * cols as usize] = f32::NAN;
            features[53 * cols as usize + 2] = f32::INFINITY;
            target[91] = f32::NEG_INFINITY;
        }
        (features, target)
    }
}
