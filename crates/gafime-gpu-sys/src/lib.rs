mod abi;
mod backend;
mod loader;
#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;
mod matrix;
mod profile;

pub use abi::{
    GafimeGpuDeviceInfoFn, GafimeGpuExecuteFn, GafimeGpuExecutionMemoryPeakFn,
    GafimeGpuGraphCapabilityFn, GafimeGpuInteractionDiagnosticsFn, GafimeGpuMatrixAllocFn,
    GafimeGpuMatrixFreeFn, GafimeGpuMatrixUpdateTargetFn, GafimeGpuMatrixUploadFn,
    GafimeGpuPermutationMemoryPeakFn, GafimeGpuPermutationPvaluesFn, GpuFunctionTable, GpuSysError,
};
pub use backend::{GpuBackend, GpuInteractionDiagnostic};
pub use loader::{CUDA_LIBRARY_ENV, METAL_LIBRARY_ENV, ROCM_LIBRARY_ENV};
#[cfg(feature = "local-cmake-experiment")]
pub use local_cmake_experiment::*;
pub use matrix::OwnedGpuMatrix;
pub use profile::{architecture_class, has_device_flag, GpuArchitectureClass, GpuDeviceProfile};

#[cfg(test)]
mod tests;
