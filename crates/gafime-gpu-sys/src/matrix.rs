use std::sync::Arc;

use gafime_orchestrator::MatrixHandle;
use libloading::Library;

use crate::abi::{status_to_gpu_result, GpuFunctionTable, GpuSysError};

pub struct OwnedGpuMatrix {
    pub(crate) handle: MatrixHandle,
    pub(crate) functions: GpuFunctionTable,
    pub(crate) library: Option<Arc<Library>>,
    #[cfg(feature = "local-cmake-experiment")]
    pub(crate) _local_cmake_experiment_owner:
        Option<Arc<crate::local_cmake_experiment::LocalCmakeExperimentDeviceStateOwner>>,
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
        // SAFETY: OwnedGpuMatrix owns a live, non-null handle from this
        // payload. The exact feature/target lengths were checked above, their
        // pointers stay live for this synchronous ABI call, and the retained
        // Library keeps the function pointer valid.
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
        // SAFETY: the matrix handle is live and owned by self, the target
        // length matches its row count, and both the slice and retained payload
        // remain live for this synchronous call.
        let status = unsafe { update_target(self.handle.raw(), target.as_ptr(), self.rows()) };
        status_to_gpu_result("gafime_gpu_matrix_update_target", status)
    }
}

impl Drop for OwnedGpuMatrix {
    fn drop(&mut self) {
        if let Some(matrix_free) = self.functions.matrix_free {
            if !self.handle.raw().is_null() {
                // SAFETY: this non-null handle was allocated by the paired
                // payload, is freed exactly once by its owner, and the retained
                // Library keeps `matrix_free` valid through this call.
                unsafe { matrix_free(self.handle.raw()) };
            }
        }
        let _keep_library_alive = &self.library;
    }
}
