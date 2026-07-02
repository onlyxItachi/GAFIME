use std::{
    env,
    error::Error,
    fmt,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    BackendKind, GafimeGpuDeviceInfo, GafimeGpuGraphCapability, GafimeGpuMatrix,
    GafimeLaunchProtocol, GafimeMatrixDesc, GafimeResultTable, GafimeStatus, GAFIME_ABI_VERSION,
    GAFIME_BACKEND_CUDA, GAFIME_BACKEND_ROCM, GAFIME_DTYPE_F32, GAFIME_MATRIX_ROW_MAJOR,
    GAFIME_RESULT_FLAG_GRAPH_REPLAYED, GAFIME_STATUS_OK,
};
use libloading::Library;

pub const CUDA_LIBRARY_ENV: &str = "GAFIME_CUDA_V1_LIB";
pub const ROCM_LIBRARY_ENV: &str = "GAFIME_ROCM_V1_LIB";

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

#[derive(Clone, Copy)]
pub struct GpuFunctionTable {
    pub device_info: Option<GafimeGpuDeviceInfoFn>,
    pub graph_capability: Option<GafimeGpuGraphCapabilityFn>,
    pub matrix_alloc: Option<GafimeGpuMatrixAllocFn>,
    pub matrix_upload: Option<GafimeGpuMatrixUploadFn>,
    pub matrix_update_target: Option<GafimeGpuMatrixUpdateTargetFn>,
    pub matrix_free: Option<GafimeGpuMatrixFreeFn>,
    pub execute: Option<GafimeGpuExecuteFn>,
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
            Self::SizeOverflow => write!(f, "GPU matrix size overflows u64 byte count"),
            Self::BackendStatus { operation, status } => {
                write!(f, "{operation} returned GPU ABI status {status}")
            }
        }
    }
}

impl Error for GpuSysError {}

#[derive(Clone)]
pub struct GpuBackend {
    kind: BackendKind,
    device_id: u32,
    functions: GpuFunctionTable,
    library: Option<Arc<Library>>,
    library_path: Option<PathBuf>,
}

impl GpuBackend {
    pub fn new(kind: BackendKind, functions: GpuFunctionTable) -> Self {
        Self {
            kind,
            device_id: 0,
            functions,
            library: None,
            library_path: None,
        }
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

    pub unsafe fn load_cuda_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_CUDA) }
    }

    pub unsafe fn load_rocm_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
    ) -> Result<Self, GpuSysError> {
        unsafe { Self::load_abi_from_path(path, device_id, GAFIME_BACKEND_ROCM) }
    }

    unsafe fn load_abi_from_path<P: AsRef<Path>>(
        path: P,
        device_id: u32,
        kind: BackendKind,
    ) -> Result<Self, GpuSysError> {
        let path = path.as_ref().to_path_buf();
        let library = Arc::new(unsafe {
            Library::new(&path).map_err(|err| GpuSysError::LoadLibrary {
                path: path.clone(),
                message: err.to_string(),
            })?
        });
        let functions = unsafe { load_function_table(&library)? };
        functions.require_complete()?;
        Ok(Self {
            kind,
            device_id,
            functions,
            library: Some(library),
            library_path: Some(path),
        })
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
        Ok(info)
    }

    pub fn graph_capability(&self) -> Result<GafimeGpuGraphCapability, GpuSysError> {
        let graph_capability = self
            .functions
            .graph_capability
            .ok_or(GpuSysError::MissingFunction("gafime_gpu_graph_capability"))?;
        let mut capability = GafimeGpuGraphCapability::default();
        let status = unsafe { graph_capability(self.device_id, &mut capability) };
        status_to_gpu_result("gafime_gpu_graph_capability", status)?;
        Ok(capability)
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
        let mut raw = ptr::null_mut();
        let status = unsafe { matrix_alloc(self.device_id, &desc, &mut raw) };
        status_to_gpu_result("gafime_gpu_matrix_alloc", status)?;
        if raw.is_null() {
            return Err(GpuSysError::InvalidInput(
                "GPU ABI returned a null matrix handle",
            ));
        }
        Ok(OwnedGpuMatrix {
            handle: MatrixHandle::native(self.kind, raw, rows, cols),
            functions: self.functions,
            library: self.library.clone(),
        })
    }
}

impl ComputeBackend for GpuBackend {
    fn backend_kind(&self) -> BackendKind {
        self.kind
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
        let status = unsafe { execute(matrix.raw(), protocol, result) };
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
}

impl OwnedGpuMatrix {
    pub fn handle(&self) -> MatrixHandle {
        self.handle
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
            .ok_or(GpuSysError::SizeOverflow)? as usize;
        if features.len() != expected_features {
            return Err(GpuSysError::InvalidInput(
                "feature buffer length does not match matrix dimensions",
            ));
        }
        if target.len() != self.rows() as usize {
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
        if target.len() != self.rows() as usize {
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
    use gafime_cpu::{matrix::CpuMatrix, CpuBackend};
    use gafime_orchestrator::{
        config::EngineConfig, execute_plan, prepare_continuous_execution, CompiledPlan,
    };
    use gafime_types::{
        GafimePermutationSchedule, GafimeRankSpec, GafimeResultTable, GAFIME_BACKEND_CPU,
        GAFIME_BACKEND_CUDA, GAFIME_BACKEND_ROCM, GAFIME_FAMILY_CONTINUOUS,
        GAFIME_GRAPH_STREAM_CAPTURE, GAFIME_LAUNCH_FLAG_GRAPH, GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
    };

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
        let backend = GpuBackend::new(
            GAFIME_BACKEND_CUDA,
            GpuFunctionTable {
                device_info: None,
                graph_capability: None,
                matrix_alloc: None,
                matrix_upload: None,
                matrix_update_target: None,
                matrix_free: None,
                execute: None,
            },
        );
        assert_eq!(backend.backend_kind(), GAFIME_BACKEND_CUDA);
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
    fn cuda_adapter_executes_when_library_is_available() {
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let info = backend.device_info().unwrap();
        assert_eq!(info.backend_kind, GAFIME_BACKEND_CUDA);

        let rows = 4;
        let cols = 2;
        let features = vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
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
    fn cuda_device_topk_returns_only_selected_rows_when_library_is_available() {
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 4;
        let cols = 3;
        let features = vec![1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
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
    fn cuda_graph_flag_replays_same_continuous_result_when_library_is_available() {
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };
        let capability = backend.graph_capability().unwrap();
        assert_eq!(capability.graph_mode, GAFIME_GRAPH_STREAM_CAPTURE);
        assert_eq!(capability.supports_device_ranking, 1);

        let rows = 32u64;
        let cols = 6u32;
        let (features, target) = parity_dataset(rows, cols);
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
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
    fn cuda_host_owns_permutation_loop_when_library_is_available() {
        let Ok(mut backend) = GpuBackend::cuda_from_env(0) else {
            return;
        };

        let rows = 32u64;
        let cols = 4u32;
        let (features, target) = parity_dataset(rows, cols);
        let matrix = backend.alloc_matrix(rows, cols).unwrap();
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
    fn cuda_mutual_info_metric_returns_finite_signal_when_library_is_available() {
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
    fn cuda_matches_cpu_for_configured_continuous_plan_arity_1_to_5() {
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
    fn rocm_host_side_topk_selects_by_primary_metric_when_available() {
        // ROCm has no device selection kernel; the host selects top-k from the
        // metrics it already copied back. Same deterministic result as CUDA/CPU.
        let Ok(mut backend) = GpuBackend::rocm_from_env(0) else {
            return;
        };

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

    fn continuous_config(backend_kind: u32) -> EngineConfig {
        let mut config = EngineConfig::default();
        config.backend_kind = backend_kind;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.permutation_tests = 0;
        config.budget.max_comb_size = 5;
        config.budget.max_combinations_per_k = 10_000;
        config
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
}
