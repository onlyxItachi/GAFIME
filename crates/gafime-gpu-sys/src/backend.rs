use std::{
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
    PrecisionComputeBackend,
};
use gafime_types::{
    BackendKind, GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimeInteractionDiagnosticBatch,
    GafimeLaunchProtocol, GafimeMatrixDesc, GafimePermutationSignificanceTable,
    GafimePermutationSignificanceTableF64, GafimePrecisionCapabilities,
    GafimePrecisionLaunchProtocol, GafimePrecisionMatrixDesc, GafimeResultTable,
    GafimeResultTableF64, PrecisionProfile, GAFIME_ABI_VERSION, GAFIME_BACKEND_CUDA,
    GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_DTYPE_F32,
    GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE, GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT, GAFIME_MATRIX_ROW_MAJOR,
    GAFIME_PRECISION_ABI_VERSION, GAFIME_RESULT_FLAG_GRAPH_REPLAYED, GAFIME_STATUS_OK,
};
use libloading::Library;

use crate::{
    abi::{status_to_gpu_result, GpuFunctionTable, GpuSysError},
    matrix::OwnedGpuMatrix,
    profile::GpuDeviceProfile,
};

#[derive(Clone)]
pub struct GpuBackend {
    pub(crate) kind: BackendKind,
    pub(crate) device_id: u32,
    pub(crate) functions: GpuFunctionTable,
    pub(crate) device_flags: u32,
    pub(crate) library: Option<Arc<Library>>,
    pub(crate) library_path: Option<PathBuf>,
    #[cfg(feature = "local-cmake-experiment")]
    pub(crate) local_cmake_experiment_lock: Option<Arc<std::sync::Mutex<()>>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuInteractionDiagnostic {
    pub overflow_row_count: u64,
    pub source_nonfinite: bool,
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
        #[cfg(feature = "local-cmake-experiment")]
        let local_cmake_experiment_lock =
            crate::local_cmake_experiment::acquire_local_cmake_experiment_lock(
                kind, device_id, &functions,
            );
        let mut backend = Self {
            kind,
            device_id,
            functions,
            device_flags: 0,
            library,
            library_path,
            #[cfg(feature = "local-cmake-experiment")]
            local_cmake_experiment_lock,
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

    fn require_legacy_matrix_abi(matrix: &MatrixHandle) -> Result<(), GpuSysError> {
        if matrix.native_abi_version() != Some(GAFIME_ABI_VERSION) {
            return Err(GpuSysError::InvalidInput(
                "legacy GPU operation requires an ABI 1.0 matrix handle",
            ));
        }
        Ok(())
    }

    fn require_precision_matrix_abi(matrix: &MatrixHandle) -> Result<(), GpuSysError> {
        if matrix.native_abi_version() != Some(GAFIME_PRECISION_ABI_VERSION) {
            return Err(GpuSysError::InvalidInput(
                "precision GPU operation requires an ABI 1.1 matrix handle",
            ));
        }
        Ok(())
    }

    pub fn device_info(&self) -> Result<GafimeGpuDeviceInfo, GpuSysError> {
        let device_info = self
            .functions
            .device_info
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_device_info"))?;
        let mut info = GafimeGpuDeviceInfo::default();
        // SAFETY: this pointer came from the trusted payload retained by self;
        // `info` is a writable local ABI value and `device_id` is plain input.
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
        // SAFETY: the retained trusted payload supplied this function pointer,
        // and `capability` is a writable local value with the exact ABI layout.
        let status = unsafe { graph_capability(self.device_id, &mut capability) };
        status_to_gpu_result("gafime_gpu_graph_capability", status)?;
        self.validate_payload_identity(capability.abi_version, capability.backend_kind)?;
        Ok(capability)
    }

    /// Return the additive ABI 1.1 precision capabilities advertised by the
    /// loaded payload. Legacy ABI 1.0 payloads deliberately fail closed here.
    pub fn precision_capabilities(&self) -> Result<GafimePrecisionCapabilities, GpuSysError> {
        let precision_capabilities =
            self.functions
                .precision_capabilities
                .ok_or(GpuSysError::MissingFunction(
                    "gafime_gpu_precision_capabilities",
                ))?;
        let mut capabilities = GafimePrecisionCapabilities::default();
        // SAFETY: the function pointer belongs to the retained trusted
        // payload and `capabilities` is a writable ABI 1.1 value.
        let status = unsafe { precision_capabilities(self.device_id, &mut capabilities) };
        status_to_gpu_result("gafime_gpu_precision_capabilities", status)?;
        if capabilities.abi_version != GAFIME_PRECISION_ABI_VERSION {
            return Err(GpuSysError::AbiVersionMismatch {
                expected: GAFIME_PRECISION_ABI_VERSION,
                actual: capabilities.abi_version,
            });
        }
        if capabilities.backend_kind != self.kind {
            return Err(GpuSysError::BackendKindMismatch {
                expected: self.kind,
                actual: capabilities.backend_kind,
            });
        }
        Ok(capabilities)
    }

    pub fn supports_precision(&self, precision: PrecisionProfile) -> Result<bool, GpuSysError> {
        let capabilities = self.precision_capabilities()?;
        let storage_mask = match precision.storage_dtype() {
            gafime_types::GAFIME_DTYPE_F32 => gafime_types::GAFIME_DTYPE_MASK_F32,
            gafime_types::GAFIME_DTYPE_F64 => gafime_types::GAFIME_DTYPE_MASK_F64,
            _ => 0,
        };
        let result_mask = match precision.result_dtype() {
            gafime_types::GAFIME_DTYPE_F32 => gafime_types::GAFIME_DTYPE_MASK_F32,
            gafime_types::GAFIME_DTYPE_F64 => gafime_types::GAFIME_DTYPE_MASK_F64,
            _ => 0,
        };
        Ok(
            (capabilities.profile_mask & precision.capability_mask()) != 0
                && (capabilities.storage_dtype_mask & storage_mask) != 0
                && (capabilities.result_dtype_mask & result_mask) != 0,
        )
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

    pub fn supports_precision_permutation_pvalues(&self, precision: PrecisionProfile) -> bool {
        match precision {
            PrecisionProfile::Fp32 => self.functions.permutation_pvalues_f32_v2.is_some(),
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
                self.functions.permutation_pvalues_f64_v2.is_some()
            }
        }
    }

    pub fn precision_permutation_profile_mask(&self) -> u32 {
        let mut mask = 0;
        if self.functions.permutation_pvalues_f32_v2.is_some() {
            mask |= PrecisionProfile::Fp32.capability_mask();
        }
        if self.functions.permutation_pvalues_f64_v2.is_some() {
            mask |= PrecisionProfile::Mixed.capability_mask()
                | PrecisionProfile::Fp64.capability_mask();
        }
        mask
    }

    pub fn supports_interaction_diagnostics(&self) -> bool {
        self.functions.interaction_diagnostics.is_some()
    }

    pub fn interaction_diagnostics(
        &self,
        matrix: &MatrixHandle,
        combo_indices: &[u32],
        max_arity: u32,
        row_count: usize,
    ) -> Result<Option<Vec<GpuInteractionDiagnostic>>, GpuSysError> {
        let Some(interaction_diagnostics) = self.functions.interaction_diagnostics else {
            return Ok(None);
        };
        if matrix.backend_kind() != self.kind || matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "matrix does not belong to the GPU diagnostics backend",
            ));
        }
        if max_arity == 0 || max_arity > 5 {
            return Err(GpuSysError::InvalidInput(
                "interaction diagnostic max_arity must be in 1..=5",
            ));
        }
        let expected = row_count
            .checked_mul(max_arity as usize)
            .ok_or(GpuSysError::SizeOverflow)?;
        if combo_indices.len() != expected {
            return Err(GpuSysError::InvalidInput(
                "interaction diagnostic combo buffer has invalid length",
            ));
        }
        let row_count_u64 = u64::try_from(row_count).map_err(|_| GpuSysError::SizeOverflow)?;
        let combo_index_count =
            u64::try_from(combo_indices.len()).map_err(|_| GpuSysError::SizeOverflow)?;
        let mut overflow_row_counts = vec![0u64; row_count];
        let mut flags = vec![0u32; row_count];
        let mut batch = GafimeInteractionDiagnosticBatch {
            abi_version: GAFIME_ABI_VERSION,
            max_arity,
            row_count: row_count_u64,
            combo_indices: combo_indices.as_ptr(),
            combo_index_count,
            overflow_row_counts: overflow_row_counts.as_mut_ptr(),
            flags: flags.as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: matrix identity/non-nullness and every batch length were
        // checked above. Input and output Vec storage remains live and uniquely
        // borrowed for this synchronous call into the retained trusted payload.
        let status = unsafe { interaction_diagnostics(matrix.raw(), &mut batch) };
        status_to_gpu_result("gafime_gpu_interaction_diagnostics", status)?;
        Ok(Some(
            overflow_row_counts
                .into_iter()
                .zip(flags)
                .map(|(overflow_row_count, flags)| GpuInteractionDiagnostic {
                    overflow_row_count,
                    source_nonfinite: (flags & GAFIME_INTERACTION_DIAGNOSTIC_FLAG_SOURCE_NONFINITE)
                        != 0,
                })
                .collect(),
        ))
    }

    pub fn supports_permutation_memory_peak(&self) -> bool {
        self.functions.permutation_memory_peak.is_some()
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

    fn negotiate_precision_launch_protocol(
        &self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
    ) -> Result<GafimeLaunchProtocol, GpuSysError> {
        Self::require_precision_matrix_abi(matrix)?;
        if matrix.backend_kind() != self.kind || matrix.raw().is_null() {
            return Err(GpuSysError::InvalidInput(
                "precision execution requires a native matrix owned by this GPU backend",
            ));
        }
        if protocol.abi_version != GAFIME_PRECISION_ABI_VERSION {
            return Err(GpuSysError::AbiVersionMismatch {
                expected: GAFIME_PRECISION_ABI_VERSION,
                actual: protocol.abi_version,
            });
        }
        if protocol.profile != matrix.precision() as u32 {
            return Err(GpuSysError::InvalidInput(
                "launch precision does not match resident matrix identity",
            ));
        }
        if protocol.base.is_null() {
            return Err(GpuSysError::InvalidInput(
                "precision launch protocol is missing its structural descriptor",
            ));
        }
        // SAFETY: callers of the synchronous ABI adapter must keep the base
        // protocol live for the wrapper borrow. Copying it here bounds native
        // use to this call and lets descriptor-generation negotiation remain
        // local without mutating caller-owned state.
        let base = unsafe { *protocol.base };
        if base.abi_version != GAFIME_ABI_VERSION {
            return Err(GpuSysError::AbiVersionMismatch {
                expected: GAFIME_ABI_VERSION,
                actual: base.abi_version,
            });
        }
        if base.backend_kind != self.kind {
            return Err(GpuSysError::BackendKindMismatch {
                expected: self.kind,
                actual: base.backend_kind,
            });
        }
        if base.n_samples != matrix.rows() || base.n_features != matrix.cols() {
            return Err(GpuSysError::InvalidInput(
                "launch dimensions do not match resident matrix identity",
            ));
        }
        Ok(self.negotiate_launch_protocol(&base))
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
        #[cfg(feature = "local-cmake-experiment")]
        let local_cmake_experiment_owner =
            crate::local_cmake_experiment::acquire_device_state_owner(
                self.kind,
                self.device_id,
                &self.functions,
                &self.library,
            );
        let mut raw = ptr::null_mut();
        // SAFETY: `matrix_alloc` belongs to the retained trusted payload,
        // `desc` is a fully initialized v1 descriptor whose byte count was
        // checked above, and `raw` is a writable local output slot.
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
            #[cfg(feature = "local-cmake-experiment")]
            _local_cmake_experiment_owner: local_cmake_experiment_owner,
        })
    }

    /// Allocate a resident matrix whose identity and element width are bound
    /// to one canonical precision profile.
    pub fn alloc_matrix_for_profile(
        &self,
        precision: PrecisionProfile,
        rows: u64,
        cols: u32,
    ) -> Result<OwnedGpuMatrix, GpuSysError> {
        if rows == 0 || cols == 0 {
            return Err(GpuSysError::InvalidInput(
                "matrix dimensions must be nonzero",
            ));
        }
        if !self.supports_precision(precision)? {
            return Err(GpuSysError::InvalidInput(
                "requested precision profile is unsupported by this GPU payload",
            ));
        }
        let element_count = rows
            .checked_mul(cols as u64)
            .ok_or(GpuSysError::SizeOverflow)?;
        let element_bytes = match precision.storage_dtype() {
            gafime_types::GAFIME_DTYPE_F32 => std::mem::size_of::<f32>() as u64,
            gafime_types::GAFIME_DTYPE_F64 => std::mem::size_of::<f64>() as u64,
            _ => {
                return Err(GpuSysError::InvalidInput(
                    "precision profile selected an unknown storage dtype",
                ));
            }
        };
        let bytes = element_count
            .checked_mul(element_bytes)
            .ok_or(GpuSysError::SizeOverflow)?;
        usize::try_from(bytes).map_err(|_| GpuSysError::SizeOverflow)?;
        let matrix_alloc = self
            .functions
            .matrix_alloc_v2
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_matrix_alloc_v2"))?;
        let desc = GafimePrecisionMatrixDesc {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: precision as u32,
            dtype: precision.storage_dtype(),
            layout: GAFIME_MATRIX_ROW_MAJOR,
            flags: 0,
            reserved32: 0,
            rows,
            cols,
            row_stride: cols,
            bytes,
            reserved: [0; 8],
        };
        #[cfg(feature = "local-cmake-experiment")]
        let local_cmake_experiment_owner =
            crate::local_cmake_experiment::acquire_device_state_owner(
                self.kind,
                self.device_id,
                &self.functions,
                &self.library,
            );
        let mut raw = ptr::null_mut();
        // SAFETY: capability negotiation completed before allocation; `desc`
        // is a fully initialized ABI 1.1 descriptor and `raw` is writable.
        let status = unsafe { matrix_alloc(self.device_id, &desc, &mut raw) };
        status_to_gpu_result("gafime_gpu_matrix_alloc_v2", status)?;
        if raw.is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU ABI returned a null matrix handle",
            ));
        }
        Ok(OwnedGpuMatrix {
            // SAFETY: the v2 payload returned a non-null matrix allocated for
            // this exact backend, profile, and shape.
            handle: unsafe {
                MatrixHandle::native_with_precision(self.kind, precision, raw, rows, cols)
            },
            functions: self.functions,
            library: self.library.clone(),
            #[cfg(feature = "local-cmake-experiment")]
            _local_cmake_experiment_owner: local_cmake_experiment_owner,
        })
    }
}

impl GpuBackend {
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
        Self::require_legacy_matrix_abi(matrix)?;
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
            // SAFETY: matrix identity/non-nullness was checked above; the
            // negotiated protocol is a live local ABI descriptor and
            // `peak_bytes` is a writable local output.
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
        // SAFETY: matrix identity/non-nullness and the row-by-metric lengths
        // were checked above. The table points only into live input slices and
        // the correctly sized output Vec for this synchronous payload call.
        let status = unsafe { permutation_pvalues(matrix.raw(), &negotiated_protocol, &mut table) };
        status_to_gpu_result("gafime_gpu_permutation_pvalues", status)?;
        Ok(Some(p_values))
    }

    pub fn permutation_pvalues_fp32_v2_with_budget(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        candidate_ids: &[u64],
        observed_metric_values: &[f32],
        metric_count: u32,
        device_budget_bytes: Option<u64>,
    ) -> Result<Option<Vec<f32>>, GpuSysError> {
        Self::require_precision_matrix_abi(matrix)?;
        let Some(permutation_pvalues) = self.functions.permutation_pvalues_f32_v2 else {
            return Ok(None);
        };
        if matrix.precision() != PrecisionProfile::Fp32 {
            return Err(GpuSysError::InvalidInput(
                "f32 significance output requires the fp32 precision profile",
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
        let negotiated_base = self.negotiate_precision_launch_protocol(matrix, protocol)?;
        let mut negotiated_protocol = *protocol;
        negotiated_protocol.base = &negotiated_base;
        if let Some(device_budget_bytes) = device_budget_bytes {
            let Some(permutation_memory_peak) = self.functions.permutation_memory_peak_v2 else {
                return Ok(None);
            };
            let mut peak_bytes = 0u64;
            // SAFETY: the matrix and precision wrapper were validated above;
            // the selected-row count and output slot are plain ABI values.
            let status = unsafe {
                permutation_memory_peak(
                    matrix.raw(),
                    &negotiated_protocol,
                    candidate_ids.len() as u64,
                    &mut peak_bytes,
                )
            };
            status_to_gpu_result("gafime_gpu_permutation_memory_peak_v2", status)?;
            if peak_bytes > device_budget_bytes {
                return Err(GpuSysError::InvalidInput(
                    "permutation p-value device-memory peak exceeds budget.vram_budget_mb",
                ));
            }
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
        // SAFETY: all matrix/protocol identities and table lengths were
        // checked; referenced slices remain live for this synchronous call.
        let status = unsafe { permutation_pvalues(matrix.raw(), &negotiated_protocol, &mut table) };
        status_to_gpu_result("gafime_gpu_permutation_pvalues_f32_v2", status)?;
        Ok(Some(p_values))
    }

    pub fn permutation_pvalues_f64_v2_with_budget(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        candidate_ids: &[u64],
        observed_metric_values: &[f64],
        metric_count: u32,
        device_budget_bytes: Option<u64>,
    ) -> Result<Option<Vec<f64>>, GpuSysError> {
        Self::require_precision_matrix_abi(matrix)?;
        let Some(permutation_pvalues) = self.functions.permutation_pvalues_f64_v2 else {
            return Ok(None);
        };
        if matrix.precision() == PrecisionProfile::Fp32 {
            return Err(GpuSysError::InvalidInput(
                "f64 significance output requires mixed or fp64 precision",
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
        let negotiated_base = self.negotiate_precision_launch_protocol(matrix, protocol)?;
        let mut negotiated_protocol = *protocol;
        negotiated_protocol.base = &negotiated_base;
        if let Some(device_budget_bytes) = device_budget_bytes {
            let Some(permutation_memory_peak) = self.functions.permutation_memory_peak_v2 else {
                return Ok(None);
            };
            let mut peak_bytes = 0u64;
            // SAFETY: the matrix and precision wrapper were validated above;
            // the selected-row count and output slot are plain ABI values.
            let status = unsafe {
                permutation_memory_peak(
                    matrix.raw(),
                    &negotiated_protocol,
                    candidate_ids.len() as u64,
                    &mut peak_bytes,
                )
            };
            status_to_gpu_result("gafime_gpu_permutation_memory_peak_v2", status)?;
            if peak_bytes > device_budget_bytes {
                return Err(GpuSysError::InvalidInput(
                    "permutation p-value device-memory peak exceeds budget.vram_budget_mb",
                ));
            }
        }
        let mut p_values = vec![f64::NAN; expected];
        let mut table = GafimePermutationSignificanceTableF64 {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            metric_count,
            row_count: candidate_ids.len() as u64,
            candidate_ids: candidate_ids.as_ptr(),
            observed_metric_values: observed_metric_values.as_ptr(),
            p_values: p_values.as_mut_ptr(),
            reserved: [0; 8],
        };
        // SAFETY: all matrix/protocol identities and table lengths were
        // checked; referenced slices remain live for this synchronous call.
        let status = unsafe { permutation_pvalues(matrix.raw(), &negotiated_protocol, &mut table) };
        status_to_gpu_result("gafime_gpu_permutation_pvalues_f64_v2", status)?;
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
        Self::require_legacy_matrix_abi(matrix).map_err(|_| {
            OrchestratorError::InvalidPlan("legacy GPU operation requires an ABI 1.0 matrix handle")
        })?;
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
        // SAFETY: matrix identity/non-nullness was checked above; the
        // orchestrator owns every pointer referenced by the live negotiated
        // protocol, and `peak_bytes` is a writable local output.
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
        Self::require_legacy_matrix_abi(matrix).map_err(|_| {
            OrchestratorError::InvalidPlan("legacy GPU operation requires an ABI 1.0 matrix handle")
        })?;
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
        // SAFETY: matrix identity/non-nullness was checked above. The
        // orchestrator keeps the protocol's storage live and the result-table
        // owner guarantees buffers matching its declared capacity and strides
        // for this synchronous call into the retained trusted payload.
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

impl PrecisionComputeBackend for GpuBackend {
    fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    fn execution_device_memory_peak_bytes_v2(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        Self::require_precision_matrix_abi(matrix).map_err(|_| {
            OrchestratorError::InvalidPlan(
                "precision GPU operation requires an ABI 1.1 matrix handle",
            )
        })?;
        let negotiated_base = self
            .negotiate_precision_launch_protocol(matrix, protocol)
            .map_err(|_| OrchestratorError::InvalidPlan("invalid precision launch protocol"))?;
        let Some(execution_memory_peak) = self.functions.execution_memory_peak_v2 else {
            return Ok(None);
        };
        let mut negotiated_protocol = *protocol;
        negotiated_protocol.base = &negotiated_base;
        let mut peak_bytes = 0u64;
        // SAFETY: the matrix and precision wrapper were validated above and
        // the output slot is writable for this synchronous payload call.
        let status =
            unsafe { execution_memory_peak(matrix.raw(), &negotiated_protocol, &mut peak_bytes) };
        if status != GAFIME_STATUS_OK {
            return Err(OrchestratorError::BackendStatus(status));
        }
        Ok(Some(peak_bytes))
    }

    fn execute_fp32(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        Self::require_precision_matrix_abi(matrix).map_err(|_| {
            OrchestratorError::InvalidPlan(
                "precision GPU operation requires an ABI 1.1 matrix handle",
            )
        })?;
        if matrix.precision() != PrecisionProfile::Fp32 {
            return Err(OrchestratorError::InvalidPlan(
                "f32 result dispatch requires the fp32 precision profile",
            ));
        }
        let execute = self
            .functions
            .execute_f32_v2
            .ok_or(OrchestratorError::Unsupported(
                "GPU payload does not implement fp32 precision",
            ))?;
        let negotiated_base = self
            .negotiate_precision_launch_protocol(matrix, protocol)
            .map_err(|_| OrchestratorError::InvalidPlan("invalid precision launch protocol"))?;
        let mut negotiated_protocol = *protocol;
        negotiated_protocol.base = &negotiated_base;
        // SAFETY: the profile-bound matrix and protocol were validated above;
        // the result owner guarantees f32 buffers matching its declared shape.
        let status = unsafe { execute(matrix.raw(), &negotiated_protocol, result) };
        if status != GAFIME_STATUS_OK {
            return Err(OrchestratorError::BackendStatus(status));
        }
        Ok(BackendExecutionStats {
            launched_chunks: negotiated_base.chunk_count as u64,
            graph_replays: u64::from((result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0),
            rows_written: result.row_count,
        })
    }

    fn execute_f64(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats> {
        Self::require_precision_matrix_abi(matrix).map_err(|_| {
            OrchestratorError::InvalidPlan(
                "precision GPU operation requires an ABI 1.1 matrix handle",
            )
        })?;
        if matrix.precision() == PrecisionProfile::Fp32 {
            return Err(OrchestratorError::InvalidPlan(
                "f64 result dispatch requires mixed or fp64 precision",
            ));
        }
        let execute = self
            .functions
            .execute_f64_v2
            .ok_or(OrchestratorError::Unsupported(
                "GPU payload does not implement f64 precision results",
            ))?;
        let negotiated_base = self
            .negotiate_precision_launch_protocol(matrix, protocol)
            .map_err(|_| OrchestratorError::InvalidPlan("invalid precision launch protocol"))?;
        let mut negotiated_protocol = *protocol;
        negotiated_protocol.base = &negotiated_base;
        // SAFETY: the profile-bound matrix and protocol were validated above;
        // the result owner guarantees f64 buffers matching its declared shape.
        let status = unsafe { execute(matrix.raw(), &negotiated_protocol, result) };
        if status != GAFIME_STATUS_OK {
            return Err(OrchestratorError::BackendStatus(status));
        }
        Ok(BackendExecutionStats {
            launched_chunks: negotiated_base.chunk_count as u64,
            graph_replays: u64::from((result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0),
            rows_written: result.row_count,
        })
    }
}
