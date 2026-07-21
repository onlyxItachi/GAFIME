pub mod arena;
pub mod decision_path;
pub mod dispatch;
pub mod kernels;
pub mod matrix;
pub mod rank;
pub mod result;
pub mod significance;
pub mod simd;
pub mod time_series;

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
};
use gafime_types::{
    GafimeArityChunk, GafimeLaunchProtocol, GafimeResultTable, GAFIME_BACKEND_CPU,
    GAFIME_FAMILY_CONTINUOUS, GAFIME_LAUNCH_FLAG_MI_APPROX,
};

use rayon::prelude::*;

use crate::kernels::{score_continuous_combo_into, ContinuousScoreScratch, MetricKernel};
use crate::matrix::CpuMatrix;

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
        if protocol.rank.top_k > 0 && protocol.rank.include_ties != 0 {
            return Err(OrchestratorError::Unsupported(
                "rank.include_ties is unsupported",
            ));
        }
        let cpu_matrix = unsafe { CpuMatrix::from_handle(matrix)? };
        validate_result_table(result, protocol)?;

        let metric_ids =
            unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
        let chunks = unsafe { slice_from_parts(protocol.chunks, protocol.chunk_count as u64)? };
        let combo_indices =
            unsafe { slice_from_parts(protocol.combo_indices.ptr, protocol.combo_indices.len)? };
        let mi_approximate = (protocol.flags & GAFIME_LAUNCH_FLAG_MI_APPROX) != 0;

        let rows_written = if protocol.rank.top_k > 0 {
            let metric_index = metric_ids
                .iter()
                .position(|&metric_id| metric_id == protocol.rank.primary_metric)
                .ok_or(OrchestratorError::InvalidPlan(
                    "rank primary metric is not in the metric set",
                ))?;
            execute_ranked_continuous(
                cpu_matrix,
                protocol,
                chunks,
                combo_indices,
                metric_ids,
                result,
                metric_index,
                mi_approximate,
            )?
        } else {
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
                    mi_approximate,
                )?;
            }
            result.row_count = rows_written;
            rows_written
        };

        Ok(BackendExecutionStats {
            launched_chunks: chunks.len() as u64,
            graph_replays: 0,
            rows_written,
        })
    }
}

/// Resolve the mutual-information bin count for a chunk from its shape hint,
/// mirroring the CUDA host (`mi_bins_for_chunk`) so CPU and GPU compute identical
/// MI. Defaults to 96 when no shape hint (or an unsupported value) is present.
fn mi_bins_for_chunk(protocol: &GafimeLaunchProtocol, chunk: &GafimeArityChunk) -> u32 {
    if protocol.shape_hints.is_null() || chunk.shape_hint_index >= protocol.shape_hint_count {
        return 96;
    }
    let hint = unsafe { &*protocol.shape_hints.add(chunk.shape_hint_index as usize) };
    match hint.vendor_hint {
        2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 => hint.vendor_hint,
        _ => 96,
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_continuous_chunk(
    matrix: &CpuMatrix,
    protocol: &GafimeLaunchProtocol,
    chunk: &GafimeArityChunk,
    combo_indices: &[u32],
    metric_ids: &[u32],
    result: &mut GafimeResultTable,
    output_row_offset: u64,
    mi_approximate: bool,
) -> OrchestratorResult<u64> {
    if chunk.family != GAFIME_FAMILY_CONTINUOUS {
        return Err(OrchestratorError::Unsupported(
            "P2 CPU checkpoint only executes continuous chunks",
        ));
    }
    let arity = chunk.arity as usize;
    let result_metric_count = result.metric_count as usize;
    let row_count = chunk.combo_count as usize;
    let combo_start = chunk.descriptor_offset as usize;
    let combo_end = combo_start.saturating_add(row_count.saturating_mul(arity));
    if combo_end > combo_indices.len() {
        return Err(OrchestratorError::InvalidPlan(
            "continuous chunk exceeds combo index buffer",
        ));
    }
    let metric_kernels = metric_ids
        .iter()
        .copied()
        .map(MetricKernel::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let mi_bins = mi_bins_for_chunk(protocol, chunk);

    // Score candidates in parallel (each is independent); rayon sizes its pool to
    // available_parallelism(). Per-worker scratch via map_init keeps the scratch
    // reuse. Writes stay serial over the order-preserving collected scores.
    let scored = (0..row_count)
        .into_par_iter()
        .map_init(ContinuousScoreScratch::default, |scratch, row| {
            let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
            // scores borrow the per-worker scratch; copy out so they outlive it.
            score_continuous_combo_into(
                matrix,
                combo,
                &metric_kernels,
                mi_bins,
                mi_approximate,
                scratch,
            )
            .map(|scores| scores.to_vec())
        })
        .collect::<OrchestratorResult<Vec<Vec<f32>>>>()?;

    for (row, scores) in scored.iter().enumerate() {
        let output_row = output_row_offset as usize + row;
        let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
        unsafe {
            write_result_row(
                result,
                protocol.max_arity as usize,
                result_metric_count,
                output_row,
                combo,
                scores,
            );
        }
    }
    Ok(row_count as u64)
}

#[allow(clippy::too_many_arguments)]
fn execute_ranked_continuous(
    matrix: &CpuMatrix,
    protocol: &GafimeLaunchProtocol,
    chunks: &[GafimeArityChunk],
    combo_indices: &[u32],
    metric_ids: &[u32],
    result: &mut GafimeResultTable,
    metric_index: usize,
    mi_approximate: bool,
) -> OrchestratorResult<u64> {
    let top_k = protocol.rank.top_k as usize;
    if top_k == 0 {
        return Ok(0);
    }
    let metric_kernels = metric_ids
        .iter()
        .copied()
        .map(MetricKernel::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let mut scratch = ContinuousScoreScratch::default();
    let mut selector = TopKSelector::new(top_k, protocol.rank.descending != 0);
    let mut global_row = 0u64;

    for chunk in chunks {
        if chunk.family != GAFIME_FAMILY_CONTINUOUS {
            return Err(OrchestratorError::Unsupported(
                "P2 CPU checkpoint only executes continuous chunks",
            ));
        }
        let arity = chunk.arity as usize;
        let row_count = chunk.combo_count as usize;
        let combo_start = chunk.descriptor_offset as usize;
        let combo_end = combo_start.saturating_add(row_count.saturating_mul(arity));
        if combo_end > combo_indices.len() {
            return Err(OrchestratorError::InvalidPlan(
                "continuous chunk exceeds combo index buffer",
            ));
        }
        let mi_bins = mi_bins_for_chunk(protocol, chunk);
        for row in 0..row_count {
            let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
            let scores = score_continuous_combo_into(
                matrix,
                combo,
                &metric_kernels,
                mi_bins,
                mi_approximate,
                &mut scratch,
            )?;
            let rank_score = *scores
                .get(metric_index)
                .ok_or(OrchestratorError::InvalidPlan(
                    "rank metric index exceeds score width",
                ))?;
            selector.consider(global_row, combo, scores, rank_score);
            global_row = global_row.saturating_add(1);
        }
    }

    let selected = selector.into_rows();
    for (rank, row) in selected.iter().enumerate() {
        unsafe {
            write_result_row_with_metadata(
                result,
                protocol.max_arity as usize,
                result.metric_count as usize,
                rank,
                &row.combo,
                &row.metrics,
                rank as u32,
                row.candidate_id,
            );
        }
    }
    result.row_count = selected.len() as u64;
    Ok(result.row_count)
}

fn validate_result_table(
    result: &GafimeResultTable,
    protocol: &GafimeLaunchProtocol,
) -> OrchestratorResult<()> {
    let planned_rows = unsafe { planned_row_count(protocol)? };
    let required_rows = if protocol.rank.top_k == 0 {
        planned_rows
    } else {
        planned_rows.min(protocol.rank.top_k as u64)
    };
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
    write_result_row_with_metadata(
        result,
        max_arity,
        metric_count,
        output_row,
        combo,
        scores,
        output_row as u32,
        output_row as u64,
    );
}

unsafe fn write_result_row_with_metadata(
    result: &mut GafimeResultTable,
    max_arity: usize,
    metric_count: usize,
    output_row: usize,
    combo: &[u32],
    scores: &[f32],
    rank: u32,
    candidate_id: u64,
) {
    let combo_base = output_row * max_arity;
    for slot in 0..max_arity {
        *result.combo_indices.add(combo_base + slot) = combo.get(slot).copied().unwrap_or(u32::MAX);
    }

    let metric_base = output_row * metric_count;
    for (index, score) in scores.iter().enumerate() {
        *result.metric_values.add(metric_base + index) = *score;
    }
    *result.ranks.add(output_row) = rank;
    *result.families.add(output_row) = GAFIME_FAMILY_CONTINUOUS;
    *result.candidate_ids.add(output_row) = candidate_id;
    *result.row_flags.add(output_row) = 0;
}

#[derive(Clone, Debug)]
struct RankedRow {
    score: f32,
    candidate_id: u64,
    combo: Vec<u32>,
    metrics: Vec<f32>,
}

#[derive(Debug)]
struct TopKSelector {
    k: usize,
    descending: bool,
    rows: Vec<RankedRow>,
}

impl TopKSelector {
    fn new(k: usize, descending: bool) -> Self {
        Self {
            k,
            descending,
            rows: Vec::with_capacity(k),
        }
    }

    fn consider(&mut self, candidate_id: u64, combo: &[u32], metrics: &[f32], score: f32) {
        if self.k == 0 || !score.is_finite() {
            return;
        }
        if self.rows.len() < self.k || self.is_better_than_worst(score, candidate_id) {
            self.rows.push(RankedRow {
                score,
                candidate_id,
                combo: combo.to_vec(),
                metrics: metrics.to_vec(),
            });
            self.sort_and_truncate();
        }
    }

    fn into_rows(mut self) -> Vec<RankedRow> {
        self.sort_and_truncate();
        self.rows
    }

    fn is_better_than_worst(&self, score: f32, candidate_id: u64) -> bool {
        self.rows
            .last()
            .map(|worst| {
                ranked_row_better(
                    score,
                    candidate_id,
                    worst.score,
                    worst.candidate_id,
                    self.descending,
                )
            })
            .unwrap_or(true)
    }

    fn sort_and_truncate(&mut self) {
        let descending = self.descending;
        self.rows.sort_by(|left, right| {
            if ranked_row_better(
                left.score,
                left.candidate_id,
                right.score,
                right.candidate_id,
                descending,
            ) {
                core::cmp::Ordering::Less
            } else if ranked_row_better(
                right.score,
                right.candidate_id,
                left.score,
                left.candidate_id,
                descending,
            ) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });
        self.rows.truncate(self.k);
    }
}

fn ranked_row_better(
    left_score: f32,
    left_candidate_id: u64,
    right_score: f32,
    right_candidate_id: u64,
    descending: bool,
) -> bool {
    let ordering = left_score
        .partial_cmp(&right_score)
        .unwrap_or(core::cmp::Ordering::Equal);
    match ordering {
        core::cmp::Ordering::Greater => descending,
        core::cmp::Ordering::Less => !descending,
        core::cmp::Ordering::Equal => left_candidate_id < right_candidate_id,
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
    fn cpu_backend_executes_every_adaptive_fixed_mi_template() {
        use gafime_orchestrator::execute_plan;
        use gafime_orchestrator::plan::combos::{
            build_continuous_plan, ContinuousPlanRequest, MI_TEMPLATE_BIN_LEVELS,
        };
        use gafime_types::{GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_METRIC_MUTUAL_INFO};

        let rows = 73_728u64;
        let feature = (0..rows)
            .map(|row| row as f32 / (rows - 1) as f32)
            .collect::<Vec<_>>();
        let target = feature
            .iter()
            .map(|&value| if value > 0.55 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let matrix = CpuMatrix::from_row_major(rows, 1, feature.clone(), target.clone()).unwrap();
        let mut backend = CpuBackend;

        for &bins in MI_TEMPLATE_BIN_LEVELS {
            let plan = build_continuous_plan(ContinuousPlanRequest {
                backend_kind: GAFIME_BACKEND_CPU,
                n_samples: rows,
                n_features: 1,
                max_arity: 1,
                max_combinations_per_arity: 1,
                metric_ids: vec![GAFIME_METRIC_MUTUAL_INFO],
                mi_bins: bins,
                rank: Default::default(),
            })
            .unwrap()
            .with_flags(GAFIME_LAUNCH_FLAG_MI_APPROX);
            let mut table = result::OwnedResultTable::new(1, 1, 1);

            execute_plan(&mut backend, &matrix.handle(), &plan, table.raw_mut()).unwrap();

            let expected = kernels::mutual_info_fixed(&feature, &target, bins);
            assert_eq!(table.metric_values()[0], expected, "bins={bins}");
        }
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
        let mut table = result::OwnedResultTable::new(2, 1, 2);
        let mut backend = CpuBackend;

        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, table.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 2);
        assert_eq!(table.raw().row_count, 2);
        assert_eq!(&table.combo_indices()[..2], &[0, 1]);
        assert_eq!(table.metric_values()[1], 1.0);
        assert_eq!(table.metric_values()[3], 1.0);
        assert_eq!(&table.candidate_ids()[..2], &[0, 1]);
    }

    #[test]
    fn cpu_backend_ranks_arity_two_scores_like_materialized_reference() {
        use gafime_orchestrator::{execute_plan, CompiledPlan};
        use gafime_types::{
            GafimeRankSpec, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
        };

        const ROWS: usize = 127;
        let columns: Vec<Vec<f32>> = (0..5)
            .map(|feature| {
                (0..ROWS)
                    .map(|row| {
                        let t = row as f32;
                        let phase = feature as f32 + 1.0;
                        (t * (0.053 * phase)).sin()
                            + (t * (0.089 + phase * 0.011)).cos() * 0.5
                            + t * (0.002 * phase)
                    })
                    .collect()
            })
            .collect();
        let means: Vec<f32> = columns
            .iter()
            .map(|column| {
                (column.iter().map(|&value| value as f64).sum::<f64>() / ROWS as f64) as f32
            })
            .collect();
        let target: Vec<f32> = (0..ROWS)
            .map(|row| {
                (columns[0][row] - means[0]) * (columns[1][row] - means[1])
                    + (columns[4][row] - means[4]) * 0.03
            })
            .collect();
        let mut features = Vec::with_capacity(ROWS * columns.len());
        for row in 0..ROWS {
            for column in &columns {
                features.push(column[row]);
            }
        }
        let matrix = CpuMatrix::from_row_major(ROWS as u64, 5, features, target).unwrap();
        let combos = vec![0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4];
        let mut expected: Vec<(u64, Vec<u32>, f32)> = combos
            .chunks_exact(2)
            .enumerate()
            .map(|(candidate_id, combo)| {
                let mut interaction = vec![1.0f32; ROWS];
                for &feature in combo {
                    let column = matrix.column(feature as usize);
                    let mean = matrix.column_mean(feature as usize);
                    for (product, &value) in interaction.iter_mut().zip(column) {
                        *product *= value - mean;
                    }
                }
                (
                    candidate_id as u64,
                    combo.to_vec(),
                    simd::r2_score(&interaction, matrix.target()),
                )
            })
            .collect();
        expected.sort_by(|left, right| {
            if ranked_row_better(left.2, left.0, right.2, right.0, true) {
                core::cmp::Ordering::Less
            } else if ranked_row_better(right.2, right.0, left.2, left.0, true) {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            ROWS as u64,
            5,
            GAFIME_FAMILY_CONTINUOUS,
            2,
            combos,
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        )
        .with_rank(GafimeRankSpec {
            top_k: 3,
            primary_metric: GAFIME_METRIC_R2,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let mut table = result::OwnedResultTable::new(3, 2, 2);
        let mut backend = CpuBackend;

        let stats = execute_plan(&mut backend, &matrix.handle(), &plan, table.raw_mut()).unwrap();

        assert_eq!(stats.rows_written, 3);
        assert_eq!(
            &table.candidate_ids()[..3],
            &[expected[0].0, expected[1].0, expected[2].0]
        );
        for (rank, expected_row) in expected.iter().take(3).enumerate() {
            assert_eq!(
                &table.combo_indices()[rank * 2..rank * 2 + 2],
                expected_row.1.as_slice()
            );
            assert!(
                (table.metric_values()[rank * 2 + 1] - expected_row.2).abs() <= 1.0e-4,
                "rank={rank}"
            );
        }
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
            mi_bins: 96,
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
