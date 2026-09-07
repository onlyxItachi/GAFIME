mod abi;
mod backend;
mod loader;
#[cfg(feature = "local-cmake-experiment")]
mod local_cmake_experiment;
mod matrix;
mod profile;
mod semantic;

pub use abi::{
    GafimeGpuDeviceInfoFn, GafimeGpuExecuteFn, GafimeGpuExecuteV2Fn,
    GafimeGpuExecutionMemoryPeakFn, GafimeGpuExecutionMemoryPeakV2Fn, GafimeGpuGraphCapabilityFn,
    GafimeGpuInteractionDiagnosticsFn, GafimeGpuInteractionDiagnosticsV2Fn, GafimeGpuMatrixAllocFn,
    GafimeGpuMatrixAllocV2Fn, GafimeGpuMatrixFreeFn, GafimeGpuMatrixFreeV2Fn,
    GafimeGpuMatrixUpdateTargetFn, GafimeGpuMatrixUpdateTargetV2Fn, GafimeGpuMatrixUploadFn,
    GafimeGpuMatrixUploadV2Fn, GafimeGpuNumericRoutesV2Fn, GafimeGpuPermutationMemoryPeakFn,
    GafimeGpuPermutationMemoryPeakV2Fn, GafimeGpuPermutationPvaluesFn,
    GafimeGpuPermutationPvaluesV2Fn, GafimeGpuSemanticBankAllocV1Fn,
    GafimeGpuSemanticBankDownloadV1Fn, GafimeGpuSemanticBankFreeV1Fn,
    GafimeGpuSemanticBankRetainV1Fn, GafimeGpuSemanticBankUploadV1Fn,
    GafimeGpuSemanticCapabilitiesV1Fn, GafimeGpuSemanticForecastV1Fn,
    GafimeGpuSemanticMaterializeV1Fn, GafimeGpuSemanticOrderedEdgeEnergyV1Fn,
    GafimeGpuSemanticPairwisePearsonV1Fn, GafimeGpuSemanticSparseGatherV1Fn, GpuFunctionTable,
    GpuSysError,
};
pub use backend::{GpuBackend, GpuInteractionDiagnostic};
pub use loader::{CUDA_LIBRARY_ENV, METAL_LIBRARY_ENV, ROCM_LIBRARY_ENV};
#[cfg(feature = "local-cmake-experiment")]
pub use local_cmake_experiment::*;
pub use matrix::OwnedGpuMatrix;
pub use profile::{architecture_class, has_device_flag, GpuArchitectureClass, GpuDeviceProfile};
pub use semantic::{
    GpuNativeEvidenceExecutor, OwnedSemanticBank, SemanticEdge, SemanticMemoryForecast,
    SemanticPearsonMode, SemanticProgramNode, SemanticScalarResult, SemanticScalarValue,
};

#[cfg(test)]
mod tests;
