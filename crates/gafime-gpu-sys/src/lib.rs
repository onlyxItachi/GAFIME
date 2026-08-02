mod abi;
mod backend;
mod loader;
#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;
mod matrix;
mod profile;

pub use abi::{
    GafimeGpuDeviceInfoFn, GafimeGpuExecuteF32V2Fn, GafimeGpuExecuteF64V2Fn, GafimeGpuExecuteFn,
    GafimeGpuExecutionMemoryPeakFn, GafimeGpuExecutionMemoryPeakV2Fn, GafimeGpuGraphCapabilityFn,
    GafimeGpuInteractionDiagnosticsFn, GafimeGpuMatrixAllocFn, GafimeGpuMatrixAllocV2Fn,
    GafimeGpuMatrixFreeFn, GafimeGpuMatrixUpdateTargetF32V2Fn, GafimeGpuMatrixUpdateTargetF64V2Fn,
    GafimeGpuMatrixUpdateTargetFn, GafimeGpuMatrixUploadF32V2Fn, GafimeGpuMatrixUploadF64V2Fn,
    GafimeGpuMatrixUploadFn, GafimeGpuPermutationMemoryPeakFn, GafimeGpuPermutationMemoryPeakV2Fn,
    GafimeGpuPermutationPvaluesF32V2Fn, GafimeGpuPermutationPvaluesF64V2Fn,
    GafimeGpuPermutationPvaluesFn, GafimeGpuPrecisionCapabilitiesFn, GpuFunctionTable, GpuSysError,
};
pub use backend::{GpuBackend, GpuInteractionDiagnostic};
pub use loader::{CUDA_LIBRARY_ENV, METAL_LIBRARY_ENV, ROCM_LIBRARY_ENV};
#[cfg(feature = "local-cmake-experiment")]
pub use local_cmake_experiment::*;
pub use matrix::OwnedGpuMatrix;
pub use profile::{architecture_class, has_device_flag, GpuArchitectureClass, GpuDeviceProfile};

#[cfg(test)]
mod tests;
