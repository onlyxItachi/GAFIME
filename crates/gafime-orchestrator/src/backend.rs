use core::ffi::c_void;

use gafime_types::{
    BackendKind, GafimeGpuMatrix, GafimeLaunchProtocol, GafimePrecisionLaunchProtocol,
    GafimeResultTable, GafimeResultTableF64, GafimeStatus, PrecisionProfile, GAFIME_ABI_VERSION,
    GAFIME_PRECISION_ABI_VERSION,
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
    precision: PrecisionProfile,
    native_abi_version: Option<u32>,
    raw: GafimeGpuMatrix,
    rows: u64,
    cols: u32,
}

impl MatrixHandle {
    pub fn host(backend_kind: BackendKind, rows: u64, cols: u32) -> Self {
        Self::host_with_precision(backend_kind, PrecisionProfile::Mixed, rows, cols)
    }

    pub fn host_with_precision(
        backend_kind: BackendKind,
        precision: PrecisionProfile,
        rows: u64,
        cols: u32,
    ) -> Self {
        Self {
            backend_kind,
            precision,
            native_abi_version: None,
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
            precision: PrecisionProfile::Mixed,
            native_abi_version: Some(GAFIME_ABI_VERSION),
            raw,
            rows,
            cols,
        }
    }

    /// Construct a profile-keyed handle for an ABI 1.1 backend matrix.
    ///
    /// # Safety
    ///
    /// `raw` must identify a live matrix allocated for `precision` by
    /// `backend_kind` and must outlive every borrow of the returned handle.
    pub unsafe fn native_with_precision(
        backend_kind: BackendKind,
        precision: PrecisionProfile,
        raw: *mut c_void,
        rows: u64,
        cols: u32,
    ) -> Self {
        Self {
            backend_kind,
            precision,
            native_abi_version: Some(GAFIME_PRECISION_ABI_VERSION),
            raw,
            rows,
            cols,
        }
    }

    pub fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    pub fn precision(&self) -> PrecisionProfile {
        self.precision
    }

    /// Return the native allocation ABI that owns this handle. Host-only
    /// handles have no native ABI identity.
    pub fn native_abi_version(&self) -> Option<u32> {
        self.native_abi_version
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

    /// Query the peak device memory for a raw ABI 1.0 launch descriptor.
    ///
    /// # Safety
    ///
    /// Every non-empty pointer/length pair reachable from `protocol` must be
    /// properly aligned, initialized, and live for this synchronous call. The
    /// pointed-to descriptor storage must not be mutated for the duration of
    /// the call.
    unsafe fn execution_device_memory_peak_bytes(
        &mut self,
        _matrix: &MatrixHandle,
        _protocol: &GafimeLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        Ok(None)
    }

    /// Execute one raw ABI 1.0 launch descriptor.
    ///
    /// # Safety
    ///
    /// In addition to the protocol requirements documented by
    /// [`Self::execution_device_memory_peak_bytes`], every non-null output
    /// pointer in `result` must reference uniquely borrowed, writable storage
    /// covering its declared capacity and strides. The matrix handle and all
    /// referenced storage must remain live for this synchronous call, and the
    /// backend must not retain any borrowed pointer after returning.
    unsafe fn execute(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats>;
}

/// Additive ABI 1.1 execution trait. Implementations dispatch once per plan to
/// a profile-specialized function table; structural planning remains shared.
pub trait PrecisionComputeBackend {
    fn backend_kind(&self) -> BackendKind;

    /// Query peak device memory for a raw ABI 1.1 precision descriptor.
    ///
    /// # Safety
    ///
    /// `protocol.base` must be null only where the operation explicitly
    /// permits it. Otherwise it must point to a live ABI 1.0 launch descriptor
    /// whose complete pointer graph satisfies the safety contract of
    /// [`ComputeBackend::execution_device_memory_peak_bytes`].
    unsafe fn execution_device_memory_peak_bytes_v2(
        &mut self,
        _matrix: &MatrixHandle,
        _protocol: &GafimePrecisionLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        Ok(None)
    }

    /// Execute a raw ABI 1.1 fp32 launch descriptor.
    ///
    /// # Safety
    ///
    /// The precision wrapper and its base descriptor must satisfy
    /// [`Self::execution_device_memory_peak_bytes_v2`]. Every output pointer in
    /// `result` must reference uniquely borrowed, writable storage covering the
    /// declared capacity and f32 strides for this synchronous call.
    unsafe fn execute_fp32(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats>;

    /// Execute a raw ABI 1.1 mixed/fp64 launch descriptor.
    ///
    /// # Safety
    ///
    /// The precision wrapper and its base descriptor must satisfy
    /// [`Self::execution_device_memory_peak_bytes_v2`]. Every output pointer in
    /// `result` must reference uniquely borrowed, writable storage covering the
    /// declared capacity and f64 strides for this synchronous call.
    unsafe fn execute_f64(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats>;
}
