pub mod arena;
pub mod dispatch;
pub mod kernels;
pub mod rank;

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{GafimeLaunchProtocol, GafimeResultTable, GAFIME_BACKEND_CPU};

#[derive(Debug, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    fn execute(
        &mut self,
        _matrix: &MatrixHandle,
        _protocol: &GafimeLaunchProtocol,
        _result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        Err(OrchestratorError::Unsupported(
            "Rust CPU kernels land in P2; P1 only freezes the backend home",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_orchestrator::ComputeBackend;

    #[test]
    fn cpu_backend_declares_cpu_kind() {
        assert_eq!(CpuBackend.backend_kind(), GAFIME_BACKEND_CPU);
    }
}
