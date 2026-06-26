pub mod arena;
pub mod dispatch;
pub mod kernels;
pub mod matrix;
pub mod rank;
pub mod result;

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    GafimeArityChunk, GafimeLaunchProtocol, GafimeResultTable, GAFIME_BACKEND_CPU,
    GAFIME_FAMILY_CONTINUOUS,
};

use crate::kernels::{score_continuous_combo, MetricKernel};
use crate::matrix::CpuMatrix;
use crate::rank::compact_result_table_top_k;

#[derive(Debug, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    fn execute(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if matrix.backend_kind() != GAFIME_BACKEND_CPU {
            return Err(OrchestratorError::InvalidPlan(
                "CPU backend received a non-CPU matrix handle",
            ));
        }
        if protocol.backend_kind != GAFIME_BACKEND_CPU {
            return Err(OrchestratorError::InvalidPlan(
                "CPU backend received a non-CPU protocol",
            ));
        }
        let cpu_matrix = unsafe { CpuMatrix::from_handle(matrix)? };
        validate_result_table(result, protocol)?;

        let metric_ids =
            unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
        let chunks = unsafe { slice_from_parts(protocol.chunks, protocol.chunk_count as u64)? };
        let combo_indices =
            unsafe { slice_from_parts(protocol.combo_indices.ptr, protocol.combo_indices.len)? };

        let mut rows_written = 0u64;
        for chunk in chunks {
            rows_written += execute_continuous_chunk(
                cpu_matrix,
                protocol,
                chunk,
                combo_indices,
                metric_ids,
                result,
                rows_written,
            )?;
        }
        result.row_count = rows_written;
        if protocol.rank.top_k > 0 {
            let metric_index = metric_ids
                .iter()
                .position(|&metric_id| metric_id == protocol.rank.primary_metric)
                .unwrap_or(0);
            rows_written = unsafe {
                compact_result_table_top_k(
                    result,
                    metric_index,
                    protocol.rank.descending != 0,
                    protocol.rank.top_k as usize,
                )?
            };
            result.row_count = rows_written;
        }

        Ok(BackendExecutionStats {
            launched_chunks: chunks.len() as u64,
            graph_replays: 0,
            rows_written,
        })
    }
}

fn execute_continuous_chunk(
    matrix: &CpuMatrix,
    protocol: &GafimeLaunchProtocol,
    chunk: &GafimeArityChunk,
    combo_indices: &[u32],
    metric_ids: &[u32],
    result: &mut GafimeResultTable,
    output_row_offset: u64,
) -> OrchestratorResult<u64> {
    if chunk.family != GAFIME_FAMILY_CONTINUOUS {
        return Err(OrchestratorError::Unsupported(
            "P2 CPU checkpoint only executes continuous chunks",
        ));
    }
    let arity = chunk.arity as usize;
    let metric_count = metric_ids.len();
    let row_count = chunk.combo_count as usize;
    let combo_start = chunk.descriptor_offset as usize;
    let combo_end = combo_start.saturating_add(row_count.saturating_mul(arity));
    if combo_end > combo_indices.len() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous chunk exceeds combo index buffer",
        ));
    }

    for row in 0..row_count {
        let output_row = output_row_offset as usize + row;
        let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
        let scores = score_continuous_combo(
            matrix,
            combo,
            metric_ids
                .iter()
                .copied()
                .map(MetricKernel::try_from)
                .collect::<Result<Vec<_>, _>>()?
                .as_slice(),
            96,
        )?;
        unsafe {
            write_result_row(
                result,
                protocol.max_arity as usize,
                metric_count,
                output_row,
                combo,
                &scores,
            );
        }
    }
    Ok(row_count as u64)
}

fn validate_result_table(
    result: &GafimeResultTable,
    protocol: &GafimeLaunchProtocol,
) -> OrchestratorResult<()> {
    let required_rows = unsafe { planned_row_count(protocol)? };
    if result.capacity < required_rows {
        return Err(OrchestratorError::InvalidPlan(
            "result table capacity is smaller than planned rows",
        ));
    }
    if result.max_arity < protocol.max_arity {
        return Err(OrchestratorError::InvalidPlan(
            "result table max arity is smaller than protocol max arity",
        ));
    }
    if result.metric_count < protocol.metric_ids.len as u32 {
        return Err(OrchestratorError::InvalidPlan(
            "result table metric capacity is smaller than protocol metric count",
        ));
    }
    if required_rows > 0
        && (result.combo_indices.is_null()
            || result.metric_values.is_null()
            || result.ranks.is_null()
            || result.families.is_null()
            || result.candidate_ids.is_null()
            || result.row_flags.is_null())
    {
        return Err(OrchestratorError::InvalidPlan(
            "result table has null output buffers",
        ));
    }
    Ok(())
}

unsafe fn planned_row_count(protocol: &GafimeLaunchProtocol) -> OrchestratorResult<u64> {
    let chunks = slice_from_parts(protocol.chunks, protocol.chunk_count as u64)?;
    Ok(chunks
        .iter()
        .fold(0u64, |total, chunk| total.saturating_add(chunk.combo_count)))
}

unsafe fn slice_from_parts<'a, T>(ptr: *const T, len: u64) -> OrchestratorResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(OrchestratorError::InvalidPlan(
            "non-empty ABI slice has null pointer",
        ));
    }
    Ok(core::slice::from_raw_parts(ptr, len as usize))
}

unsafe fn write_result_row(
    result: &mut GafimeResultTable,
    max_arity: usize,
    metric_count: usize,
    output_row: usize,
    combo: &[u32],
    scores: &[f32],
) {
    let combo_base = output_row * max_arity;
    for slot in 0..max_arity {
        *result.combo_indices.add(combo_base + slot) = combo.get(slot).copied().unwrap_or(u32::MAX);
    }

    let metric_base = output_row * metric_count;
    for (index, score) in scores.iter().enumerate() {
        *result.metric_values.add(metric_base + index) = *score;
    }
    *result.ranks.add(output_row) = output_row as u32;
    *result.families.add(output_row) = GAFIME_FAMILY_CONTINUOUS;
    *result.candidate_ids.add(output_row) = output_row as u64;
    *result.row_flags.add(output_row) = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_orchestrator::ComputeBackend;

    #[test]
    fn cpu_backend_declares_cpu_kind() {
        assert_eq!(CpuBackend.backend_kind(), GAFIME_BACKEND_CPU);
    }

    #[test]
    fn cpu_backend_executes_continuous_result_table() {
        use gafime_orchestrator::{execute_plan, CompiledPlan};
        use gafime_types::{GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};

        let matrix = CpuMatrix::from_row_major(
            4,
            3,
            vec![1.0, 2.0, 0.5, 2.0, 1.0, 1.5, 3.0, 0.0, 2.5, 4.0, -1.0, 3.5],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let handle = matrix.handle();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            4,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );
        let mut table = result::OwnedResultTable::new(3, 1, 2);
        let mut backend = CpuBackend;

        let stats = execute_plan(&mut backend, &handle, &plan, table.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 3);
        assert_eq!(table.raw().row_count, 3);
        assert!((table.metric_values()[0] - 1.0).abs() < 1e-6);
        assert!((table.metric_values()[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_backend_honors_rank_top_k() {
        use gafime_orchestrator::{execute_plan, CompiledPlan};
        use gafime_types::{
            GafimeRankSpec, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
        };

        let matrix = CpuMatrix::from_row_major(
            5,
            3,
            vec![
                1.0, 5.0, 1.0, 2.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 2.0, 1.0, 5.0, 1.0, 1.0,
            ],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            5,
            3,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 2,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut table = result::OwnedResultTable::new(3, 1, 2);
        let mut backend = CpuBackend;

        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, table.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 2);
        assert_eq!(table.raw().row_count, 2);
        assert_eq!(&table.combo_indices()[..2], &[0, 1]);
        assert_eq!(table.metric_values()[1], 1.0);
        assert_eq!(table.metric_values()[3], 1.0);
    }

    #[test]
    fn cpu_backend_executes_mixed_arity_continuous_plan() {
        use gafime_orchestrator::execute_plan;
        use gafime_orchestrator::plan::combos::{build_continuous_plan, ContinuousPlanRequest};
        use gafime_types::{GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};

        let matrix = CpuMatrix::from_row_major(
            4,
            3,
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let plan = build_continuous_plan(ContinuousPlanRequest {
            backend_kind: GAFIME_BACKEND_CPU,
            n_samples: 4,
            n_features: 3,
            max_arity: 2,
            max_combinations_per_arity: 16,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            rank: Default::default(),
        })
        .unwrap();
        let planned_rows: u64 = plan.chunks().iter().map(|chunk| chunk.combo_count).sum();
        let mut table = result::OwnedResultTable::new(planned_rows, 2, 2);
        let mut backend = CpuBackend;

        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, table.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 6);
        assert_eq!(table.raw().row_count, 6);
        assert_eq!(
            &table.combo_indices()[..8],
            &[0, u32::MAX, 1, u32::MAX, 2, u32::MAX, 0, 1]
        );
    }
}
