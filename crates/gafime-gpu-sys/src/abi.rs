use std::{error::Error, fmt, path::PathBuf};

use gafime_types::{
    BackendKind, GafimeConstBufferView, GafimeGpuDeviceInfo, GafimeGpuGraphCapability,
    GafimeGpuMatrix, GafimeGpuSemanticBank, GafimeInteractionDiagnosticBatch, GafimeLaunchProtocol,
    GafimeMatrixDesc, GafimeMutableBufferView, GafimeNumericInteractionDiagnosticBatch,
    GafimeNumericLaunchProtocol, GafimeNumericMatrixDesc, GafimeNumericResultTable,
    GafimeNumericRoute, GafimeNumericSignificanceTable, GafimePermutationSignificanceTable,
    GafimeResultTable, GafimeSemanticBankDesc, GafimeSemanticCapabilities,
    GafimeSemanticEdgeEnergyBatch, GafimeSemanticForecastRequest, GafimeSemanticMemoryForecast,
    GafimeSemanticPearsonBatch, GafimeSemanticProgramBatch, GafimeSemanticScalarResultTable,
    GafimeSemanticSparseGatherBatch, GafimeSliceU32, GafimeStatus, PrecisionProfile,
    GAFIME_STATUS_OK,
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
pub type GafimeGpuInteractionDiagnosticsFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    diagnostics: *mut GafimeInteractionDiagnosticBatch,
) -> GafimeStatus;
pub type GafimeGpuNumericRoutesV2Fn = unsafe extern "C" fn(
    device_id: u32,
    consumer_abi_version: u32,
    route_stride: u32,
    routes_out: *mut GafimeNumericRoute,
    route_capacity: u32,
    route_count_out: *mut u32,
) -> GafimeStatus;
pub type GafimeGpuMatrixAllocV2Fn = unsafe extern "C" fn(
    device_id: u32,
    matrix_desc: *const GafimeNumericMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus;
pub type GafimeGpuMatrixUploadV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    route: *const GafimeNumericRoute,
    features: *const GafimeConstBufferView,
    target: *const GafimeConstBufferView,
    rows: u64,
    cols: u32,
) -> GafimeStatus;
pub type GafimeGpuMatrixUpdateTargetV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    route: *const GafimeNumericRoute,
    target: *const GafimeConstBufferView,
    rows: u64,
) -> GafimeStatus;
pub type GafimeGpuExecuteV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    result_out: *mut GafimeNumericResultTable,
) -> GafimeStatus;
pub type GafimeGpuExecutionMemoryPeakV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    peak_bytes_out: *mut u64,
) -> GafimeStatus;
pub type GafimeGpuPermutationMemoryPeakV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    selected_row_count: u64,
    peak_bytes_out: *mut u64,
) -> GafimeStatus;
pub type GafimeGpuPermutationPvaluesV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeNumericLaunchProtocol,
    significance_out: *mut GafimeNumericSignificanceTable,
) -> GafimeStatus;
pub type GafimeGpuInteractionDiagnosticsV2Fn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    diagnostics: *mut GafimeNumericInteractionDiagnosticBatch,
) -> GafimeStatus;
pub type GafimeGpuMatrixFreeV2Fn = unsafe extern "C" fn(matrix: GafimeGpuMatrix) -> GafimeStatus;

/// Optional all-or-nothing resident semantic-arithmetic table.  These
/// callbacks carry only typed physical slots; Rust retains semantic identity,
/// context and policy above this ABI boundary.
pub type GafimeGpuSemanticCapabilitiesV1Fn = unsafe extern "C" fn(
    device_id: u32,
    consumer_abi_version: u32,
    capabilities_out: *mut GafimeSemanticCapabilities,
) -> GafimeStatus;
pub type GafimeGpuSemanticBankAllocV1Fn = unsafe extern "C" fn(
    device_id: u32,
    desc: *const GafimeSemanticBankDesc,
    bank_out: *mut GafimeGpuSemanticBank,
) -> GafimeStatus;
pub type GafimeGpuSemanticBankUploadV1Fn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    route: *const GafimeNumericRoute,
    source_columns: *const GafimeConstBufferView,
) -> GafimeStatus;
pub type GafimeGpuSemanticMaterializeV1Fn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticProgramBatch,
) -> GafimeStatus;
pub type GafimeGpuSemanticPairwisePearsonV1Fn = unsafe extern "C" fn(
    left_bank: GafimeGpuSemanticBank,
    right_bank: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticPearsonBatch,
    results_out: *mut GafimeSemanticScalarResultTable,
) -> GafimeStatus;
pub type GafimeGpuSemanticOrderedEdgeEnergyV1Fn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticEdgeEnergyBatch,
    results_out: *mut GafimeSemanticScalarResultTable,
) -> GafimeStatus;
pub type GafimeGpuSemanticSparseGatherV1Fn = unsafe extern "C" fn(
    source_bank: GafimeGpuSemanticBank,
    destination_bank: GafimeGpuSemanticBank,
    batch: *const GafimeSemanticSparseGatherBatch,
) -> GafimeStatus;
pub type GafimeGpuSemanticForecastV1Fn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    request: *const GafimeSemanticForecastRequest,
    forecast_out: *mut GafimeSemanticMemoryForecast,
) -> GafimeStatus;
pub type GafimeGpuSemanticBankRetainV1Fn = unsafe extern "C" fn(
    source_bank: GafimeGpuSemanticBank,
    slots: GafimeSliceU32,
    retained_bank_out: *mut GafimeGpuSemanticBank,
) -> GafimeStatus;
pub type GafimeGpuSemanticBankDownloadV1Fn = unsafe extern "C" fn(
    bank: GafimeGpuSemanticBank,
    slots: GafimeSliceU32,
    route: *const GafimeNumericRoute,
    columns_out: *mut GafimeMutableBufferView,
) -> GafimeStatus;
pub type GafimeGpuSemanticBankFreeV1Fn =
    unsafe extern "C" fn(bank: GafimeGpuSemanticBank) -> GafimeStatus;
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
    pub interaction_diagnostics: Option<GafimeGpuInteractionDiagnosticsFn>,
    pub numeric_routes_v2: Option<GafimeGpuNumericRoutesV2Fn>,
    pub matrix_alloc_v2: Option<GafimeGpuMatrixAllocV2Fn>,
    pub matrix_upload_v2: Option<GafimeGpuMatrixUploadV2Fn>,
    pub matrix_update_target_v2: Option<GafimeGpuMatrixUpdateTargetV2Fn>,
    pub execute_v2: Option<GafimeGpuExecuteV2Fn>,
    pub execution_memory_peak_v2: Option<GafimeGpuExecutionMemoryPeakV2Fn>,
    pub permutation_memory_peak_v2: Option<GafimeGpuPermutationMemoryPeakV2Fn>,
    pub permutation_pvalues_v2: Option<GafimeGpuPermutationPvaluesV2Fn>,
    pub interaction_diagnostics_v2: Option<GafimeGpuInteractionDiagnosticsV2Fn>,
    pub matrix_free_v2: Option<GafimeGpuMatrixFreeV2Fn>,
    pub semantic_capabilities_v1: Option<GafimeGpuSemanticCapabilitiesV1Fn>,
    pub semantic_bank_alloc_v1: Option<GafimeGpuSemanticBankAllocV1Fn>,
    pub semantic_bank_upload_v1: Option<GafimeGpuSemanticBankUploadV1Fn>,
    pub semantic_materialize_v1: Option<GafimeGpuSemanticMaterializeV1Fn>,
    pub semantic_pairwise_pearson_v1: Option<GafimeGpuSemanticPairwisePearsonV1Fn>,
    pub semantic_ordered_edge_energy_v1: Option<GafimeGpuSemanticOrderedEdgeEnergyV1Fn>,
    pub semantic_sparse_gather_v1: Option<GafimeGpuSemanticSparseGatherV1Fn>,
    pub semantic_forecast_v1: Option<GafimeGpuSemanticForecastV1Fn>,
    pub semantic_bank_retain_v1: Option<GafimeGpuSemanticBankRetainV1Fn>,
    pub semantic_bank_download_v1: Option<GafimeGpuSemanticBankDownloadV1Fn>,
    pub semantic_bank_free_v1: Option<GafimeGpuSemanticBankFreeV1Fn>,
    #[cfg(feature = "local-cmake-experiment")]
    pub local_cmake_experiment: crate::local_cmake_experiment::LocalCmakeExperimentFunctions,
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

    fn has_any_precision_surface(&self) -> bool {
        self.numeric_routes_v2.is_some()
            || self.matrix_alloc_v2.is_some()
            || self.matrix_upload_v2.is_some()
            || self.matrix_update_target_v2.is_some()
            || self.execute_v2.is_some()
            || self.execution_memory_peak_v2.is_some()
            || self.permutation_memory_peak_v2.is_some()
            || self.permutation_pvalues_v2.is_some()
            || self.interaction_diagnostics_v2.is_some()
            || self.matrix_free_v2.is_some()
    }

    fn has_any_semantic_surface(&self) -> bool {
        self.semantic_capabilities_v1.is_some()
            || self.semantic_bank_alloc_v1.is_some()
            || self.semantic_bank_upload_v1.is_some()
            || self.semantic_materialize_v1.is_some()
            || self.semantic_pairwise_pearson_v1.is_some()
            || self.semantic_ordered_edge_energy_v1.is_some()
            || self.semantic_sparse_gather_v1.is_some()
            || self.semantic_forecast_v1.is_some()
            || self.semantic_bank_retain_v1.is_some()
            || self.semantic_bank_download_v1.is_some()
            || self.semantic_bank_free_v1.is_some()
    }

    /// Validate the optional semantic table as one indivisible contract.  An
    /// older payload with no symbols remains usable for its frozen ABI routes;
    /// a partial semantic table fails closed rather than becoming an accidental
    /// per-operation fallback surface.
    pub(crate) fn require_semantic_common(&self) -> Result<(), GpuSysError> {
        if self.semantic_capabilities_v1.is_none() {
            if self.has_any_semantic_surface() {
                return Err(GpuSysError::MissingFunction(
                    "gafime_gpu_semantic_capabilities_v1",
                ));
            }
            return Err(GpuSysError::SemanticAbiUnavailable);
        }
        if self.semantic_bank_alloc_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_bank_alloc_v1",
            ));
        }
        if self.semantic_bank_upload_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_bank_upload_v1",
            ));
        }
        if self.semantic_materialize_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_materialize_v1",
            ));
        }
        if self.semantic_pairwise_pearson_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_pairwise_pearson_v1",
            ));
        }
        if self.semantic_ordered_edge_energy_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_ordered_edge_energy_v1",
            ));
        }
        if self.semantic_sparse_gather_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_sparse_gather_v1",
            ));
        }
        if self.semantic_forecast_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_forecast_v1",
            ));
        }
        if self.semantic_bank_retain_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_bank_retain_v1",
            ));
        }
        if self.semantic_bank_download_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_bank_download_v1",
            ));
        }
        if self.semantic_bank_free_v1.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_semantic_bank_free_v1",
            ));
        }
        Ok(())
    }

    /// Validate the complete ABI 1.1 operation table shared by every
    /// canonical precision profile. A payload that advertises
    /// `numeric_routes_v2` must export all ten generic symbols; this check runs
    /// before a typed allocation callback can be selected. The function table
    /// keeps dynamic-loader results as `Option` because symbol probing is
    /// inherently fallible and legacy ABI 1.0 tables do not carry this v2
    /// surface, but a partial v2 table is never a supported fallback.
    pub(crate) fn require_precision_common(&self) -> Result<(), GpuSysError> {
        if self.numeric_routes_v2.is_none() {
            if self.has_any_precision_surface() {
                return Err(GpuSysError::MissingFunction("gafime_gpu_numeric_routes_v2"));
            }
            return Err(GpuSysError::PrecisionAbiUnavailable);
        }
        if self.matrix_alloc_v2.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_alloc_v2"));
        }
        if self.matrix_upload_v2.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_upload_v2"));
        }
        if self.matrix_update_target_v2.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_matrix_update_target_v2",
            ));
        }
        if self.execute_v2.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_execute_v2"));
        }
        if self.execution_memory_peak_v2.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_execution_memory_peak_v2",
            ));
        }
        if self.permutation_memory_peak_v2.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_permutation_memory_peak_v2",
            ));
        }
        if self.permutation_pvalues_v2.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_permutation_pvalues_v2",
            ));
        }
        if self.interaction_diagnostics_v2.is_none() {
            return Err(GpuSysError::MissingFunction(
                "gafime_gpu_interaction_diagnostics_v2",
            ));
        }
        if self.matrix_free_v2.is_none() {
            return Err(GpuSysError::MissingFunction("gafime_gpu_matrix_free_v2"));
        }
        Ok(())
    }

    /// Validate the canonical operation table needed by one advertised route.
    pub(crate) fn require_precision_profile(
        &self,
        precision: PrecisionProfile,
    ) -> Result<(), GpuSysError> {
        let _ = precision;
        self.require_precision_common()
    }
}

#[derive(Clone, Debug)]
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
    PrecisionAbiUnavailable,
    SemanticAbiUnavailable,
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
            Self::PrecisionAbiUnavailable => write!(
                f,
                "canonical precision profiles require GPU ABI 1.1; the loaded payload exposes only legacy ABI 1.0"
            ),
            Self::SemanticAbiUnavailable => write!(
                f,
                "loaded GPU payload does not expose the optional semantic-arithmetic ABI"
            ),
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

/// Load the complete v1 function table from a live GAFIME GPU payload.
///
/// # Safety
///
/// `library` must be a trusted GAFIME GPU payload whose exported symbol names
/// and function signatures match the v1 C ABI. The returned pointers may only
/// be called while `library` remains loaded.
pub(crate) unsafe fn load_function_table(
    library: &Library,
) -> Result<GpuFunctionTable, GpuSysError> {
    // SAFETY: the caller guarantees that this is a trusted v1 payload and that
    // every requested symbol has the function-pointer type declared by the
    // shared C ABI. GpuBackend retains the Library for every pointer's lifetime.
    Ok(unsafe {
        GpuFunctionTable {
            device_info: Some(load_symbol::<GafimeGpuDeviceInfoFn>(
                library,
                "gafime_gpu_device_info",
            )?),
            graph_capability: Some(load_symbol::<GafimeGpuGraphCapabilityFn>(
                library,
                "gafime_gpu_graph_capability",
            )?),
            matrix_alloc: Some(load_symbol::<GafimeGpuMatrixAllocFn>(
                library,
                "gafime_gpu_matrix_alloc",
            )?),
            matrix_upload: Some(load_symbol::<GafimeGpuMatrixUploadFn>(
                library,
                "gafime_gpu_matrix_upload",
            )?),
            matrix_update_target: Some(load_symbol::<GafimeGpuMatrixUpdateTargetFn>(
                library,
                "gafime_gpu_matrix_update_target",
            )?),
            matrix_free: Some(load_symbol::<GafimeGpuMatrixFreeFn>(
                library,
                "gafime_gpu_matrix_free",
            )?),
            execute: Some(load_symbol::<GafimeGpuExecuteFn>(
                library,
                "gafime_gpu_execute",
            )?),
            execution_memory_peak: load_optional_symbol::<GafimeGpuExecutionMemoryPeakFn>(
                library,
                "gafime_gpu_execution_memory_peak",
            ),
            permutation_memory_peak: load_optional_symbol::<GafimeGpuPermutationMemoryPeakFn>(
                library,
                "gafime_gpu_permutation_memory_peak",
            ),
            permutation_pvalues: load_optional_symbol::<GafimeGpuPermutationPvaluesFn>(
                library,
                "gafime_gpu_permutation_pvalues",
            ),
            interaction_diagnostics: load_optional_symbol::<GafimeGpuInteractionDiagnosticsFn>(
                library,
                "gafime_gpu_interaction_diagnostics",
            ),
            numeric_routes_v2: load_optional_symbol::<GafimeGpuNumericRoutesV2Fn>(
                library,
                "gafime_gpu_numeric_routes_v2",
            ),
            matrix_alloc_v2: load_optional_symbol::<GafimeGpuMatrixAllocV2Fn>(
                library,
                "gafime_gpu_matrix_alloc_v2",
            ),
            matrix_upload_v2: load_optional_symbol::<GafimeGpuMatrixUploadV2Fn>(
                library,
                "gafime_gpu_matrix_upload_v2",
            ),
            matrix_update_target_v2: load_optional_symbol::<GafimeGpuMatrixUpdateTargetV2Fn>(
                library,
                "gafime_gpu_matrix_update_target_v2",
            ),
            execute_v2: load_optional_symbol::<GafimeGpuExecuteV2Fn>(
                library,
                "gafime_gpu_execute_v2",
            ),
            execution_memory_peak_v2: load_optional_symbol::<GafimeGpuExecutionMemoryPeakV2Fn>(
                library,
                "gafime_gpu_execution_memory_peak_v2",
            ),
            permutation_memory_peak_v2: load_optional_symbol::<GafimeGpuPermutationMemoryPeakV2Fn>(
                library,
                "gafime_gpu_permutation_memory_peak_v2",
            ),
            permutation_pvalues_v2: load_optional_symbol::<GafimeGpuPermutationPvaluesV2Fn>(
                library,
                "gafime_gpu_permutation_pvalues_v2",
            ),
            interaction_diagnostics_v2: load_optional_symbol::<GafimeGpuInteractionDiagnosticsV2Fn>(
                library,
                "gafime_gpu_interaction_diagnostics_v2",
            ),
            matrix_free_v2: load_optional_symbol::<GafimeGpuMatrixFreeV2Fn>(
                library,
                "gafime_gpu_matrix_free_v2",
            ),
            semantic_capabilities_v1: load_optional_symbol::<GafimeGpuSemanticCapabilitiesV1Fn>(
                library,
                "gafime_gpu_semantic_capabilities_v1",
            ),
            semantic_bank_alloc_v1: load_optional_symbol::<GafimeGpuSemanticBankAllocV1Fn>(
                library,
                "gafime_gpu_semantic_bank_alloc_v1",
            ),
            semantic_bank_upload_v1: load_optional_symbol::<GafimeGpuSemanticBankUploadV1Fn>(
                library,
                "gafime_gpu_semantic_bank_upload_v1",
            ),
            semantic_materialize_v1: load_optional_symbol::<GafimeGpuSemanticMaterializeV1Fn>(
                library,
                "gafime_gpu_semantic_materialize_v1",
            ),
            semantic_pairwise_pearson_v1: load_optional_symbol::<
                GafimeGpuSemanticPairwisePearsonV1Fn,
            >(
                library, "gafime_gpu_semantic_pairwise_pearson_v1"
            ),
            semantic_ordered_edge_energy_v1: load_optional_symbol::<
                GafimeGpuSemanticOrderedEdgeEnergyV1Fn,
            >(
                library,
                "gafime_gpu_semantic_ordered_edge_energy_v1",
            ),
            semantic_sparse_gather_v1: load_optional_symbol::<GafimeGpuSemanticSparseGatherV1Fn>(
                library,
                "gafime_gpu_semantic_sparse_gather_v1",
            ),
            semantic_forecast_v1: load_optional_symbol::<GafimeGpuSemanticForecastV1Fn>(
                library,
                "gafime_gpu_semantic_forecast_v1",
            ),
            semantic_bank_retain_v1: load_optional_symbol::<GafimeGpuSemanticBankRetainV1Fn>(
                library,
                "gafime_gpu_semantic_bank_retain_v1",
            ),
            semantic_bank_download_v1: load_optional_symbol::<GafimeGpuSemanticBankDownloadV1Fn>(
                library,
                "gafime_gpu_semantic_bank_download_v1",
            ),
            semantic_bank_free_v1: load_optional_symbol::<GafimeGpuSemanticBankFreeV1Fn>(
                library,
                "gafime_gpu_semantic_bank_free_v1",
            ),
            #[cfg(feature = "local-cmake-experiment")]
            local_cmake_experiment: crate::local_cmake_experiment::load_function_table(library),
        }
    })
}

/// Load a required symbol using the function-pointer type declared by the ABI.
///
/// # Safety
///
/// `symbol` must identify an export whose native signature is exactly `T`, and
/// `library` must remain loaded for every use of the copied pointer.
unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &'static str) -> Result<T, GpuSysError> {
    // SAFETY: the caller binds this symbol name to its exact v1 ABI function
    // type and keeps the source Library alive for the copied pointer's lifetime.
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

/// Load an optional symbol using the function-pointer type declared by the ABI.
///
/// # Safety
///
/// When present, `symbol` must have the exact native signature `T`, and
/// `library` must remain loaded for every use of the copied pointer.
pub(crate) unsafe fn load_optional_symbol<T: Copy>(
    library: &Library,
    symbol: &'static str,
) -> Option<T> {
    // SAFETY: the caller binds this optional symbol name to its exact v1 ABI
    // function type and keeps the source Library alive. Absence maps to None.
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
