use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    BackendKind, GafimeGpuDeviceInfo, GafimeGpuMatrix, GafimeLaunchProtocol, GafimeMatrixDesc,
    GafimeResultTable, GafimeStatus,
};

pub type GafimeGpuDeviceInfoFn =
    unsafe extern "C" fn(device_id: u32, info_out: *mut GafimeGpuDeviceInfo) -> GafimeStatus;
pub type GafimeGpuMatrixAllocFn = unsafe extern "C" fn(
    device_id: u32,
    matrix_desc: *const GafimeMatrixDesc,
    matrix_out: *mut GafimeGpuMatrix,
) -> GafimeStatus;
pub type GafimeGpuExecuteFn = unsafe extern "C" fn(
    matrix: GafimeGpuMatrix,
    protocol: *const GafimeLaunchProtocol,
    result_out: *mut GafimeResultTable,
) -> GafimeStatus;

#[derive(Clone, Copy)]
pub struct GpuFunctionTable {
    pub device_info: Option<GafimeGpuDeviceInfoFn>,
    pub matrix_alloc: Option<GafimeGpuMatrixAllocFn>,
    pub execute: Option<GafimeGpuExecuteFn>,
}

#[derive(Clone, Copy)]
pub struct GpuBackend {
    kind: BackendKind,
    functions: GpuFunctionTable,
}

impl GpuBackend {
    pub fn new(kind: BackendKind, functions: GpuFunctionTable) -> Self {
        Self { kind, functions }
    }
}

impl ComputeBackend for GpuBackend {
    fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    fn execute(
        &mut self,
        _matrix: &MatrixHandle,
        _protocol: &GafimeLaunchProtocol,
        _result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if self.functions.execute.is_none() {
            return Err(OrchestratorError::Unsupported(
                "GPU C ABI payload is not linked in the P1 skeleton",
            ));
        }
        Err(OrchestratorError::Unsupported(
            "GPU execution adapter is wired in P3 with resident buffers and graph replay",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::GAFIME_BACKEND_CUDA;

    #[test]
    fn gpu_backend_declares_vendor_kind() {
        let backend = GpuBackend::new(
            GAFIME_BACKEND_CUDA,
            GpuFunctionTable {
                device_info: None,
                matrix_alloc: None,
                execute: None,
            },
        );
        assert_eq!(backend.backend_kind(), GAFIME_BACKEND_CUDA);
    }
}
