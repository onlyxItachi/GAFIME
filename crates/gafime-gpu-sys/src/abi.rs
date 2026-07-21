use std::{error::Error, fmt, path::PathBuf};

use gafime_types::{
    BackendKind, GafimeDecisionPathBatch, GafimeDecisionPathScoreBatch, GafimeGpuDeviceInfo,
    GafimeGpuGraphCapability, GafimeGpuMatrix, GafimeLaunchProtocol, GafimeMatrixDesc,
    GafimePermutationSignificanceTable, GafimeResultTable, GafimeStatus, GAFIME_STATUS_OK,
};
use libloading::Library;

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
pub type GafimeGpuPermutationMemoryPeakFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    selected_row_count: u64,
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
    pub permutation_memory_peak: Option<GafimeGpuPermutationMemoryPeakFn>,
    pub permutation_pvalues: Option<GafimeGpuPermutationPvaluesFn>,
    pub decision_path_membership: Option<GafimeGpuDecisionPathMembershipFn>,
    pub decision_path_score: Option<GafimeGpuDecisionPathScoreFn>,
    pub decision_path_release_device_state: Option<GafimeGpuDecisionPathReleaseDeviceStateFn>,
}

impl GpuFunctionTable {
    pub(crate) fn require_complete(&self) -> Result<(), GpuSysError> {
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
pub(crate) unsafe fn load_function_table(
    library: &Library,
) -> Result<GpuFunctionTable, GpuSysError> {
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
        permutation_memory_peak: unsafe {
            load_optional_symbol::<GafimeGpuPermutationMemoryPeakFn>(
                library,
                "gafime_gpu_permutation_memory_peak",
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

pub(crate) fn status_to_gpu_result(
    operation: &'static str,
    status: GafimeStatus,
) -> Result<(), GpuSysError> {
    if status == GAFIME_STATUS_OK {
        Ok(())
    } else {
        Err(GpuSysError::BackendStatus { operation, status })
    }
}
