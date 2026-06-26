pub mod backend;
pub mod cache;
pub mod plan;
pub mod reduce;
pub mod schedule;

pub use backend::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
pub use plan::CompiledPlan;

use gafime_types::GafimeResultTable;

pub fn execute_plan<B: ComputeBackend>(
    backend: &mut B,
    matrix: &MatrixHandle,
    plan: &CompiledPlan,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    plan.validate()?;
    backend.execute(matrix, plan.protocol(), result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{GAFIME_BACKEND_CPU, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_PEARSON};

    #[test]
    fn execute_plan_validates_before_backend_call() {
        struct CountingBackend {
            calls: usize,
        }

        impl ComputeBackend for CountingBackend {
            fn backend_kind(&self) -> u32 {
                GAFIME_BACKEND_CPU
            }

            fn execute(
                &mut self,
                _matrix: &MatrixHandle,
                _protocol: &gafime_types::GafimeLaunchProtocol,
                _result: &mut GafimeResultTable,
            ) -> OrchestratorResult<BackendExecutionStats> {
                self.calls += 1;
                Ok(BackendExecutionStats::default())
            }
        }

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            32,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_PEARSON],
        );
        let mut backend = CountingBackend { calls: 0 };
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 32, 3);
        let mut result = GafimeResultTable::default();

        execute_plan(&mut backend, &matrix, &plan, &mut result).unwrap();
        assert_eq!(backend.calls, 1);
    }
}
