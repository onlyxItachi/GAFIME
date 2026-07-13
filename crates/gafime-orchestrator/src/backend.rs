use core::ffi::c_void;

use gafime_types::{
    BackendKind, GafimeGpuMatrix, GafimeLaunchProtocol, GafimeResultTable, GafimeStatus,
};

pub type OrchestratorResult<T> = Result<T, OrchestratorError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorError {
    InvalidPlan(&'static str),
    Unsupported(&'static str),
    BackendStatus(GafimeStatus),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendExecutionStats {
    pub launched_chunks: u64,
    pub graph_replays: u64,
    pub rows_written: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct MatrixHandle {
    backend_kind: BackendKind,
    raw: GafimeGpuMatrix,
    rows: u64,
    cols: u32,
}

impl MatrixHandle {
    pub fn host(backend_kind: BackendKind, rows: u64, cols: u32) -> Self {
        Self {
            backend_kind,
            raw: core::ptr::null_mut(),
            rows,
            cols,
        }
    }

    /// Construct a handle for a backend-owned native matrix.
    ///
    /// # Safety
    ///
    /// `raw` must identify a live matrix owned by `backend_kind`, with the
    /// supplied dimensions. The caller must also ensure that the native matrix
    /// outlives every borrow of the returned handle.
    ///
    /// ```compile_fail
    /// use gafime_orchestrator::MatrixHandle;
    /// use gafime_types::GAFIME_BACKEND_CUDA;
    ///
    /// let _ = MatrixHandle::native(
    ///     GAFIME_BACKEND_CUDA,
    ///     core::ptr::null_mut(),
    ///     1,
    ///     1,
    /// );
    /// ```
    pub unsafe fn native(
        backend_kind: BackendKind,
        raw: *mut c_void,
        rows: u64,
        cols: u32,
    ) -> Self {
        Self {
            backend_kind,
            raw,
            rows,
            cols,
        }
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    pub fn raw(&self) -> GafimeGpuMatrix {
        self.raw
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn cols(&self) -> u32 {
        self.cols
    }
}

pub trait ComputeBackend {
    fn backend_kind(&self) -> BackendKind;

    fn execute(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats>;
}
