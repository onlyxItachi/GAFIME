mod abi;
mod backend;
mod loader;
mod matrix;
mod profile;

pub use abi::{
    GafimeGpuDecisionPathMembershipFn, GafimeGpuDecisionPathReleaseDeviceStateFn,
    GafimeGpuDecisionPathScoreFn, GafimeGpuDeviceInfoFn, GafimeGpuExecuteFn,
    GafimeGpuExecutionMemoryPeakFn, GafimeGpuGraphCapabilityFn, GafimeGpuInteractionDiagnosticsFn,
    GafimeGpuMatrixAllocFn, GafimeGpuMatrixFreeFn, GafimeGpuMatrixUpdateTargetFn,
    GafimeGpuMatrixUploadFn, GafimeGpuPermutationMemoryPeakFn, GafimeGpuPermutationPvaluesFn,
    GpuFunctionTable, GpuSysError,
};
pub use backend::{GpuBackend, GpuInteractionDiagnostic};
pub use loader::{CUDA_LIBRARY_ENV, METAL_LIBRARY_ENV, ROCM_LIBRARY_ENV};
pub use matrix::OwnedGpuMatrix;
pub use profile::{
    architecture_class, has_device_flag, DecisionPathRtPolicy, GpuArchitectureClass,
    GpuDeviceProfile,
};

#[cfg(test)]
mod tests;
