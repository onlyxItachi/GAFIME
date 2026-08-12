pub mod arena;
pub mod decision_path;
pub mod diagnostics;
pub mod dispatch;
pub mod kernels;
pub mod matrix;
pub mod precision;
pub mod rank;
pub mod result;
pub mod significance;
pub mod simd;
pub mod time_series;

use gafime_orchestrator::{
    BackendExecutionStats, ComputeBackend, MatrixHandle, OrchestratorError, OrchestratorResult,
    PrecisionComputeBackend,
};
use gafime_types::{
    GafimeArityChunk, GafimeLaunchProtocol, GafimePrecisionLaunchProtocol, GafimeResultTable,
    GafimeResultTableF64, PrecisionProfile, GAFIME_BACKEND_CPU, GAFIME_FAMILY_CONTINUOUS,
    GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_PRECISION_ABI_VERSION,
};

use rayon::prelude::*;

use crate::kernels::{
    precision::{score_precision_continuous_combo_into, PrecisionScoreScratch},
    score_continuous_combo_into, ContinuousScoreScratch, MetricKernel,
};
use crate::matrix::CpuMatrix;
use crate::precision::{CpuPrecisionMatrix, CpuPrecisionSlice};

#[derive(Debug, Default)]
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    unsafe fn execute(
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
        // SAFETY: backend_kind was checked above. CPU handles are created only
        // by CpuMatrix::handle, whose borrow keeps the matrix alive and whose
        // embedded shape is validated again by from_handle.
        let cpu_matrix = unsafe { CpuMatrix::from_handle(matrix)? };
        validate_result_table(result, protocol)?;

        // SAFETY: the prepared execution plan owns each protocol buffer and
        // keeps it live and immutable for this synchronous backend call.
        let metric_ids =
            unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
        // SAFETY: the prepared execution plan owns `chunk_count` initialized
        // chunk descriptors and keeps them live for this call.
        let chunks = unsafe { slice_from_parts(protocol.chunks, protocol.chunk_count as u64)? };
        // SAFETY: the prepared execution plan owns the declared combo-index
        // buffer and keeps it live and immutable for this call.
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

/// ABI 1.1 Core execution.  The profile is selected once at this trait
/// boundary, then the f32 and f64 routines below contain separate typed loops.
/// The historical ABI 1.0 [`ComputeBackend`] remains intact for legacy callers.
impl PrecisionComputeBackend for CpuBackend {
    fn backend_kind(&self) -> u32 {
        GAFIME_BACKEND_CPU
    }

    unsafe fn execution_device_memory_peak_bytes_v2(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        let base = precision_base_protocol(matrix, protocol)?;
        // Core owns host resident storage rather than device memory.  Validate
        // the typed protocol above, then report the exact GPU-facing peak of
        // zero instead of pretending f32/f64 host buffers are device bytes.
        let _ = base;
        Ok(Some(0))
    }

    unsafe fn execute_fp32(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if matrix.precision() != PrecisionProfile::Fp32 {
            return Err(OrchestratorError::InvalidPlan(
                "f32 Core result dispatch requires the fp32 profile",
            ));
        }
        let base = precision_base_protocol(matrix, protocol)?;
        // SAFETY: precision_base_protocol confirmed this is a CPU handle; the
        // owner-bound CpuPrecisionMatrix handle keeps the matrix live for this
        // synchronous execution and validates its profile/shape again here.
        let cpu_matrix = unsafe { CpuPrecisionMatrix::from_handle(matrix)? };
        execute_precision_fp32(cpu_matrix, base, result)
    }

    unsafe fn execute_f64(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut GafimeResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if matrix.precision() == PrecisionProfile::Fp32 {
            return Err(OrchestratorError::InvalidPlan(
                "f64 Core result dispatch requires mixed or fp64 precision",
            ));
        }
        let base = precision_base_protocol(matrix, protocol)?;
        // SAFETY: as in execute_fp32, the typed resident handle is checked and
        // remains live for this synchronous Core call.
        let cpu_matrix = unsafe { CpuPrecisionMatrix::from_handle(matrix)? };
        execute_precision_f64(cpu_matrix, base, result)
    }
}

fn precision_base_protocol<'a>(
    matrix: &MatrixHandle,
    protocol: &'a GafimePrecisionLaunchProtocol,
) -> OrchestratorResult<&'a GafimeLaunchProtocol> {
    if matrix.backend_kind() != GAFIME_BACKEND_CPU {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision backend received a non-CPU matrix handle",
        ));
    }
    if protocol.abi_version != GAFIME_PRECISION_ABI_VERSION {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision launch protocol ABI version is unsupported",
        ));
    }
    if protocol.profile != matrix.precision() as u32 {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision launch profile does not match resident matrix identity",
        ));
    }
    // SAFETY: the prepared execution owner retains its base protocol for the
    // synchronous backend call. A null pointer is rejected before dereference.
    let base = unsafe { protocol.base.as_ref() }.ok_or(OrchestratorError::InvalidPlan(
        "CPU precision launch protocol is missing its structural descriptor",
    ))?;
    if base.abi_version != gafime_types::GAFIME_ABI_VERSION {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision base protocol ABI version is unsupported",
        ));
    }
    if base.backend_kind != GAFIME_BACKEND_CPU {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision base protocol targets a non-CPU backend",
        ));
    }
    if base.n_samples != matrix.rows() || base.n_features != matrix.cols() {
        return Err(OrchestratorError::InvalidPlan(
            "CPU precision base protocol shape does not match resident matrix",
        ));
    }
    Ok(base)
}

fn execute_precision_fp32(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    validate_result_table(result, protocol)?;
    // SAFETY: the prepared plan owns its metric-id buffer throughout this
    // synchronous execution and validation above binds this protocol to Core.
    let metric_ids = unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
    // SAFETY: the prepared plan owns `chunk_count` initialized descriptors for
    // this synchronous execution.
    let chunks = unsafe { slice_from_parts(protocol.chunks, protocol.chunk_count as u64)? };
    // SAFETY: the prepared plan owns its immutable combo index buffer for this
    // synchronous execution.
    let combo_indices =
        unsafe { slice_from_parts(protocol.combo_indices.ptr, protocol.combo_indices.len)? };
    let metric_kernels = metric_ids
        .iter()
        .copied()
        .map(MetricKernel::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let mi_approximate = (protocol.flags & GAFIME_LAUNCH_FLAG_MI_APPROX) != 0;
    if protocol.rank.top_k > 0 && protocol.rank.include_ties != 0 {
        return Err(OrchestratorError::Unsupported(
            "rank.include_ties is unsupported",
        ));
    }

    let rows_written = if protocol.rank.top_k == 0 {
        let mut output_row = 0usize;
        for chunk in chunks {
            output_row += execute_precision_chunk_fp32(
                matrix,
                protocol,
                chunk,
                combo_indices,
                &metric_kernels,
                mi_approximate,
                result,
                output_row,
            )?;
        }
        result.row_count = output_row as u64;
        result.row_count
    } else {
        execute_precision_ranked_fp32(
            matrix,
            protocol,
            chunks,
            combo_indices,
            &metric_kernels,
            mi_approximate,
            result,
        )?
    };
    Ok(BackendExecutionStats {
        launched_chunks: chunks.len() as u64,
        graph_replays: 0,
        rows_written,
    })
}

fn execute_precision_f64(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    result: &mut GafimeResultTableF64,
) -> OrchestratorResult<BackendExecutionStats> {
    validate_result_table_f64(result, protocol)?;
    // SAFETY: the prepared plan owns its metric-id buffer throughout this
    // synchronous execution and validation above binds this protocol to Core.
    let metric_ids = unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
    // SAFETY: the prepared plan owns `chunk_count` initialized descriptors for
    // this synchronous execution.
    let chunks = unsafe { slice_from_parts(protocol.chunks, protocol.chunk_count as u64)? };
    // SAFETY: the prepared plan owns its immutable combo index buffer for this
    // synchronous execution.
    let combo_indices =
        unsafe { slice_from_parts(protocol.combo_indices.ptr, protocol.combo_indices.len)? };
    let metric_kernels = metric_ids
        .iter()
        .copied()
        .map(MetricKernel::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let mi_approximate = (protocol.flags & GAFIME_LAUNCH_FLAG_MI_APPROX) != 0;
    if protocol.rank.top_k > 0 && protocol.rank.include_ties != 0 {
        return Err(OrchestratorError::Unsupported(
            "rank.include_ties is unsupported",
        ));
    }

    let rows_written = if protocol.rank.top_k == 0 {
        let mut output_row = 0usize;
        for chunk in chunks {
            output_row += execute_precision_chunk_f64(
                matrix,
                protocol,
                chunk,
                combo_indices,
                &metric_kernels,
                mi_approximate,
                result,
                output_row,
            )?;
        }
        result.row_count = output_row as u64;
        result.row_count
    } else {
        execute_precision_ranked_f64(
            matrix,
            protocol,
            chunks,
            combo_indices,
            &metric_kernels,
            mi_approximate,
            result,
        )?
    };
    Ok(BackendExecutionStats {
        launched_chunks: chunks.len() as u64,
        graph_replays: 0,
        rows_written,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_precision_chunk_fp32(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    chunk: &GafimeArityChunk,
    combo_indices: &[u32],
    metric_kernels: &[MetricKernel],
    mi_approximate: bool,
    result: &mut GafimeResultTable,
    output_offset: usize,
) -> OrchestratorResult<usize> {
    let (arity, row_count, combo_start) = validated_precision_chunk(chunk, combo_indices)?;
    let mi_bins = mi_bins_for_chunk(protocol, chunk);
    let metric_count = precision_metric_count(metric_kernels)?;
    let score_count = row_count
        .checked_mul(metric_count)
        .ok_or(OrchestratorError::InvalidPlan(
            "fp32 precision score buffer length overflows",
        ))?;
    // One contiguous, owned result buffer is shared safely by disjoint Rayon
    // chunks.  Each worker owns its reusable scratch; raw ABI writes remain
    // below in deterministic plan order.
    let mut scored = vec![0.0f32; score_count];
    scored
        .par_chunks_mut(metric_count)
        .enumerate()
        .try_for_each_init(
            || PrecisionScoreScratch::new(PrecisionProfile::Fp32),
            |scratch, (row, destination)| {
                let combo =
                    &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
                let CpuPrecisionSlice::F32(scores) = score_precision_continuous_combo_into(
                    matrix,
                    combo,
                    metric_kernels,
                    mi_bins,
                    mi_approximate,
                    scratch,
                )?
                else {
                    return Err(OrchestratorError::InvalidPlan(
                        "fp32 Core profile produced a non-f32 score row",
                    ));
                };
                if scores.len() != destination.len() {
                    return Err(OrchestratorError::InvalidPlan(
                        "fp32 precision score width does not match its metrics",
                    ));
                }
                destination.copy_from_slice(scores);
                #[cfg(test)]
                precision_parallelism_test_hook::record_candidate_worker();
                Ok(())
            },
        )?;

    for (row, scores) in scored.chunks_exact(metric_count).enumerate() {
        let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
        let candidate_id = chunk.combo_row_offset.checked_add(row as u64).ok_or(
            OrchestratorError::InvalidPlan("fp32 precision candidate id exceeds the ABI range"),
        )?;
        // SAFETY: validate_result_table validated all result strides/capacity;
        // row is bounded by the current planned chunk.
        unsafe {
            write_precision_result_row_f32(
                result,
                result.max_arity as usize,
                result.metric_count as usize,
                output_offset + row,
                combo,
                scores,
                (output_offset + row) as u32,
                candidate_id,
            );
        }
    }
    Ok(row_count)
}

#[allow(clippy::too_many_arguments)]
fn execute_precision_chunk_f64(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    chunk: &GafimeArityChunk,
    combo_indices: &[u32],
    metric_kernels: &[MetricKernel],
    mi_approximate: bool,
    result: &mut GafimeResultTableF64,
    output_offset: usize,
) -> OrchestratorResult<usize> {
    let (arity, row_count, combo_start) = validated_precision_chunk(chunk, combo_indices)?;
    let mi_bins = mi_bins_for_chunk(protocol, chunk);
    let metric_count = precision_metric_count(metric_kernels)?;
    let score_count = row_count
        .checked_mul(metric_count)
        .ok_or(OrchestratorError::InvalidPlan(
            "f64 precision score buffer length overflows",
        ))?;
    // As in the fp32 path, Rayon writes only disjoint owned score chunks.
    // The raw ABI table is populated serially after all candidate work ends.
    let mut scored = vec![0.0f64; score_count];
    scored
        .par_chunks_mut(metric_count)
        .enumerate()
        .try_for_each_init(
            || PrecisionScoreScratch::new(matrix.profile()),
            |scratch, (row, destination)| {
                let combo =
                    &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
                let CpuPrecisionSlice::F64(scores) = score_precision_continuous_combo_into(
                    matrix,
                    combo,
                    metric_kernels,
                    mi_bins,
                    mi_approximate,
                    scratch,
                )?
                else {
                    return Err(OrchestratorError::InvalidPlan(
                        "mixed/fp64 Core profile produced a non-f64 score row",
                    ));
                };
                if scores.len() != destination.len() {
                    return Err(OrchestratorError::InvalidPlan(
                        "f64 precision score width does not match its metrics",
                    ));
                }
                destination.copy_from_slice(scores);
                #[cfg(test)]
                precision_parallelism_test_hook::record_candidate_worker();
                Ok(())
            },
        )?;

    for (row, scores) in scored.chunks_exact(metric_count).enumerate() {
        let combo = &combo_indices[combo_start + row * arity..combo_start + (row + 1) * arity];
        let candidate_id = chunk.combo_row_offset.checked_add(row as u64).ok_or(
            OrchestratorError::InvalidPlan("f64 precision candidate id exceeds the ABI range"),
        )?;
        // SAFETY: validate_result_table_f64 validated all result
        // strides/capacity; row is bounded by the planned chunk.
        unsafe {
            write_precision_result_row_f64(
                result,
                result.max_arity as usize,
                result.metric_count as usize,
                output_offset + row,
                combo,
                scores,
                (output_offset + row) as u32,
                candidate_id,
            );
        }
    }
    Ok(row_count)
}

fn precision_metric_count(metric_kernels: &[MetricKernel]) -> OrchestratorResult<usize> {
    if metric_kernels.is_empty() {
        return Err(OrchestratorError::InvalidPlan(
            "precision execution requires at least one metric",
        ));
    }
    Ok(metric_kernels.len())
}

fn validated_precision_chunk(
    chunk: &GafimeArityChunk,
    combo_indices: &[u32],
) -> OrchestratorResult<(usize, usize, usize)> {
    if chunk.family != GAFIME_FAMILY_CONTINUOUS {
        return Err(OrchestratorError::Unsupported(
            "Core precision execution receives materialized continuous chunks only",
        ));
    }
    let arity = chunk.arity as usize;
    let row_count = chunk.combo_count as usize;
    let combo_start = chunk.descriptor_offset as usize;
    let combo_end = combo_start.saturating_add(row_count.saturating_mul(arity));
    if combo_end > combo_indices.len() {
        return Err(OrchestratorError::InvalidPlan(
            "precision continuous chunk exceeds combo index buffer",
        ));
    }
    Ok((arity, row_count, combo_start))
}

#[derive(Clone, Copy, Debug)]
struct PrecisionCandidateWork {
    combo_offset: usize,
    arity: usize,
    mi_bins: u32,
    candidate_id: u64,
}

#[derive(Clone, Copy, Debug)]
struct PrecisionRankedRowF32 {
    score: f32,
    candidate_id: u64,
    combo_offset: usize,
    arity: usize,
    score_offset: usize,
}

#[derive(Clone, Copy, Debug)]
struct PrecisionRankedRowF64 {
    score: f64,
    candidate_id: u64,
    combo_offset: usize,
    arity: usize,
    score_offset: usize,
}

fn precision_ranked_work(
    protocol: &GafimeLaunchProtocol,
    chunks: &[GafimeArityChunk],
    combo_indices: &[u32],
) -> OrchestratorResult<Vec<PrecisionCandidateWork>> {
    let mut work = Vec::new();
    for chunk in chunks {
        let (arity, row_count, combo_start) = validated_precision_chunk(chunk, combo_indices)?;
        let mi_bins = mi_bins_for_chunk(protocol, chunk);
        for row in 0..row_count {
            let candidate_id = chunk.combo_row_offset.checked_add(row as u64).ok_or(
                OrchestratorError::InvalidPlan("precision candidate id exceeds the ABI range"),
            )?;
            work.push(PrecisionCandidateWork {
                combo_offset: combo_start + row * arity,
                arity,
                mi_bins,
                candidate_id,
            });
        }
    }
    Ok(work)
}

#[allow(clippy::too_many_arguments)]
fn execute_precision_ranked_fp32(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    chunks: &[GafimeArityChunk],
    combo_indices: &[u32],
    metric_kernels: &[MetricKernel],
    mi_approximate: bool,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<u64> {
    // SAFETY: the prepared plan keeps its metric-id slice live and immutable
    // while this synchronous precision ranking call runs.
    let metric_ids = unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
    let metric_index = metric_ids
        .iter()
        .position(|&metric| metric == protocol.rank.primary_metric)
        .ok_or(OrchestratorError::InvalidPlan(
            "rank primary metric is not in the metric set",
        ))?;
    let metric_count = precision_metric_count(metric_kernels)?;
    let work = precision_ranked_work(protocol, chunks, combo_indices)?;
    let score_count =
        work.len()
            .checked_mul(metric_count)
            .ok_or(OrchestratorError::InvalidPlan(
                "fp32 precision ranked score buffer length overflows",
            ))?;
    let mut scored = vec![0.0f32; score_count];
    work.par_iter()
        .zip(scored.par_chunks_mut(metric_count))
        .try_for_each_init(
            || PrecisionScoreScratch::new(PrecisionProfile::Fp32),
            |scratch, (candidate, destination)| {
                let combo = &combo_indices
                    [candidate.combo_offset..candidate.combo_offset + candidate.arity];
                let CpuPrecisionSlice::F32(scores) = score_precision_continuous_combo_into(
                    matrix,
                    combo,
                    metric_kernels,
                    candidate.mi_bins,
                    mi_approximate,
                    scratch,
                )?
                else {
                    return Err(OrchestratorError::InvalidPlan(
                        "fp32 Core profile produced a non-f32 ranking score row",
                    ));
                };
                if scores.len() != destination.len() {
                    return Err(OrchestratorError::InvalidPlan(
                        "fp32 precision ranking score width does not match its metrics",
                    ));
                }
                destination.copy_from_slice(scores);
                #[cfg(test)]
                precision_parallelism_test_hook::record_candidate_worker();
                Ok(())
            },
        )?;

    let mut rows = Vec::with_capacity(work.len());
    for (index, candidate) in work.iter().enumerate() {
        let score_offset = index * metric_count;
        let score = *scored[score_offset..score_offset + metric_count]
            .get(metric_index)
            .ok_or(OrchestratorError::InvalidPlan(
                "rank metric index exceeds score width",
            ))?;
        if score.is_finite() {
            rows.push(PrecisionRankedRowF32 {
                score,
                candidate_id: candidate.candidate_id,
                combo_offset: candidate.combo_offset,
                arity: candidate.arity,
                score_offset,
            });
        }
    }
    let descending = protocol.rank.descending != 0;
    rows.sort_by(|left, right| rank_order_f32(left, right, descending));
    rows.truncate(protocol.rank.top_k as usize);
    for (rank, row) in rows.iter().enumerate() {
        // SAFETY: result validation bounds output by top_k and validates every
        // typed result pointer. Row contents came from validated protocols.
        unsafe {
            write_precision_result_row_f32(
                result,
                result.max_arity as usize,
                result.metric_count as usize,
                rank,
                &combo_indices[row.combo_offset..row.combo_offset + row.arity],
                &scored[row.score_offset..row.score_offset + metric_count],
                rank as u32,
                row.candidate_id,
            );
        }
    }
    result.row_count = rows.len() as u64;
    Ok(result.row_count)
}

#[allow(clippy::too_many_arguments)]
fn execute_precision_ranked_f64(
    matrix: &CpuPrecisionMatrix,
    protocol: &GafimeLaunchProtocol,
    chunks: &[GafimeArityChunk],
    combo_indices: &[u32],
    metric_kernels: &[MetricKernel],
    mi_approximate: bool,
    result: &mut GafimeResultTableF64,
) -> OrchestratorResult<u64> {
    // SAFETY: the prepared plan keeps its metric-id slice live and immutable
    // while this synchronous precision ranking call runs.
    let metric_ids = unsafe { slice_from_parts(protocol.metric_ids.ptr, protocol.metric_ids.len)? };
    let metric_index = metric_ids
        .iter()
        .position(|&metric| metric == protocol.rank.primary_metric)
        .ok_or(OrchestratorError::InvalidPlan(
            "rank primary metric is not in the metric set",
        ))?;
    let metric_count = precision_metric_count(metric_kernels)?;
    let work = precision_ranked_work(protocol, chunks, combo_indices)?;
    let score_count =
        work.len()
            .checked_mul(metric_count)
            .ok_or(OrchestratorError::InvalidPlan(
                "f64 precision ranked score buffer length overflows",
            ))?;
    let mut scored = vec![0.0f64; score_count];
    work.par_iter()
        .zip(scored.par_chunks_mut(metric_count))
        .try_for_each_init(
            || PrecisionScoreScratch::new(matrix.profile()),
            |scratch, (candidate, destination)| {
                let combo = &combo_indices
                    [candidate.combo_offset..candidate.combo_offset + candidate.arity];
                let CpuPrecisionSlice::F64(scores) = score_precision_continuous_combo_into(
                    matrix,
                    combo,
                    metric_kernels,
                    candidate.mi_bins,
                    mi_approximate,
                    scratch,
                )?
                else {
                    return Err(OrchestratorError::InvalidPlan(
                        "mixed/fp64 Core profile produced a non-f64 ranking score row",
                    ));
                };
                if scores.len() != destination.len() {
                    return Err(OrchestratorError::InvalidPlan(
                        "f64 precision ranking score width does not match its metrics",
                    ));
                }
                destination.copy_from_slice(scores);
                #[cfg(test)]
                precision_parallelism_test_hook::record_candidate_worker();
                Ok(())
            },
        )?;

    let mut rows = Vec::with_capacity(work.len());
    for (index, candidate) in work.iter().enumerate() {
        let score_offset = index * metric_count;
        let score = *scored[score_offset..score_offset + metric_count]
            .get(metric_index)
            .ok_or(OrchestratorError::InvalidPlan(
                "rank metric index exceeds score width",
            ))?;
        if score.is_finite() {
            rows.push(PrecisionRankedRowF64 {
                score,
                candidate_id: candidate.candidate_id,
                combo_offset: candidate.combo_offset,
                arity: candidate.arity,
                score_offset,
            });
        }
    }
    let descending = protocol.rank.descending != 0;
    rows.sort_by(|left, right| rank_order_f64(left, right, descending));
    rows.truncate(protocol.rank.top_k as usize);
    for (rank, row) in rows.iter().enumerate() {
        // SAFETY: f64 table validation bounds output by top_k and validates its
        // exact f64 result pointer; no f32 staging is used.
        unsafe {
            write_precision_result_row_f64(
                result,
                result.max_arity as usize,
                result.metric_count as usize,
                rank,
                &combo_indices[row.combo_offset..row.combo_offset + row.arity],
                &scored[row.score_offset..row.score_offset + metric_count],
                rank as u32,
                row.candidate_id,
            );
        }
    }
    result.row_count = rows.len() as u64;
    Ok(result.row_count)
}

fn rank_order_f32(
    left: &PrecisionRankedRowF32,
    right: &PrecisionRankedRowF32,
    descending: bool,
) -> core::cmp::Ordering {
    let ordering = left
        .score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal);
    let ordering = if descending {
        ordering.reverse()
    } else {
        ordering
    };
    ordering.then(left.candidate_id.cmp(&right.candidate_id))
}

fn rank_order_f64(
    left: &PrecisionRankedRowF64,
    right: &PrecisionRankedRowF64,
    descending: bool,
) -> core::cmp::Ordering {
    let ordering = left
        .score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal);
    let ordering = if descending {
        ordering.reverse()
    } else {
        ordering
    };
    ordering.then(left.candidate_id.cmp(&right.candidate_id))
}

fn validate_result_table_f64(
    result: &GafimeResultTableF64,
    protocol: &GafimeLaunchProtocol,
) -> OrchestratorResult<()> {
    // SAFETY: the prepared protocol owns `chunk_count` initialized chunk
    // descriptors for this synchronous validation pass.
    let planned_rows = unsafe { planned_row_count(protocol)? };
    let required_rows = if protocol.rank.top_k == 0 {
        planned_rows
    } else {
        planned_rows.min(protocol.rank.top_k as u64)
    };
    if result.capacity < required_rows {
        return Err(OrchestratorError::InvalidPlan(
            "f64 result table capacity is smaller than planned rows",
        ));
    }
    if result.max_arity < protocol.max_arity {
        return Err(OrchestratorError::InvalidPlan(
            "f64 result table max arity is smaller than protocol max arity",
        ));
    }
    if result.metric_count < protocol.metric_ids.len as u32 {
        return Err(OrchestratorError::InvalidPlan(
            "f64 result table metric capacity is smaller than protocol metric count",
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
            "f64 result table has null output buffers",
        ));
    }
    Ok(())
}

/// Write a typed ABI 1.1 fp32 row.
///
/// # Safety
///
/// The validated ABI table owns writable buffers for `output_row` at the
/// supplied strides, and `combo`/`scores` are no wider than those strides.
#[allow(clippy::too_many_arguments)]
unsafe fn write_precision_result_row_f32(
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
        // SAFETY: caller validates the result row window and this loop remains
        // within max_arity slots for the selected output row.
        unsafe {
            *result.combo_indices.add(combo_base + slot) =
                combo.get(slot).copied().unwrap_or(u32::MAX);
        }
    }
    let metric_base = output_row * metric_count;
    for index in 0..metric_count {
        // SAFETY: caller validates metric stride; scores are checked against
        // the protocol metric width by the profile-specialized scorer.
        unsafe {
            *result.metric_values.add(metric_base + index) =
                scores.get(index).copied().unwrap_or(0.0);
        }
    }
    // SAFETY: each metadata pointer was validated with the same row capacity.
    unsafe {
        *result.ranks.add(output_row) = rank;
        *result.families.add(output_row) = GAFIME_FAMILY_CONTINUOUS;
        *result.candidate_ids.add(output_row) = candidate_id;
        *result.row_flags.add(output_row) = 0;
    }
}

/// Write a typed ABI 1.1 f64 row with no intermediate fp32 quantization.
///
/// # Safety
///
/// The validated ABI table owns writable buffers for `output_row` at the
/// supplied strides, and `combo`/`scores` are no wider than those strides.
#[allow(clippy::too_many_arguments)]
unsafe fn write_precision_result_row_f64(
    result: &mut GafimeResultTableF64,
    max_arity: usize,
    metric_count: usize,
    output_row: usize,
    combo: &[u32],
    scores: &[f64],
    rank: u32,
    candidate_id: u64,
) {
    let combo_base = output_row * max_arity;
    for slot in 0..max_arity {
        // SAFETY: caller validates the result row window and this loop remains
        // within max_arity slots for the selected output row.
        unsafe {
            *result.combo_indices.add(combo_base + slot) =
                combo.get(slot).copied().unwrap_or(u32::MAX);
        }
    }
    let metric_base = output_row * metric_count;
    for index in 0..metric_count {
        // SAFETY: this is the ABI 1.1 `*mut f64` typed result surface.
        unsafe {
            *result.metric_values.add(metric_base + index) =
                scores.get(index).copied().unwrap_or(0.0);
        }
    }
    // SAFETY: each metadata pointer was validated with the same row capacity.
    unsafe {
        *result.ranks.add(output_row) = rank;
        *result.families.add(output_row) = GAFIME_FAMILY_CONTINUOUS;
        *result.candidate_ids.add(output_row) = candidate_id;
        *result.row_flags.add(output_row) = 0;
    }
}

/// Resolve the mutual-information bin count for a chunk from its shape hint,
/// mirroring the CUDA host (`mi_bins_for_chunk`) so CPU and GPU compute identical
/// MI. Defaults to 96 when no shape hint (or an unsupported value) is present.
fn mi_bins_for_chunk(protocol: &GafimeLaunchProtocol, chunk: &GafimeArityChunk) -> u32 {
    if protocol.shape_hints.is_null() || chunk.shape_hint_index >= protocol.shape_hint_count {
        return 96;
    }
    // SAFETY: nullness and the shape-hint index bound were checked above; the
    // prepared plan owns this initialized array for the protocol's lifetime.
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
        // SAFETY: validate_result_table proved the result buffers cover every
        // planned row and declared stride. This row is bounded by its validated
        // chunk and combo/scores do not exceed max_arity/metric_count.
        unsafe {
            write_result_row(
                result,
                result.max_arity as usize,
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
        // SAFETY: validate_result_table bounded selected rows by top_k and
        // proved all result buffers cover the declared capacity and strides.
        // Selector rows originate from validated combo and metric slices.
        unsafe {
            write_result_row_with_metadata(
                result,
                result.max_arity as usize,
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
    // SAFETY: protocol chunks belong to the prepared plan and remain live for
    // this validation call; planned_row_count only reads their combo counts.
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

/// Sum the rows declared by a protocol's chunk descriptors.
///
/// # Safety
///
/// For a nonzero `chunk_count`, `protocol.chunks` must point to that many
/// initialized, properly aligned `GafimeArityChunk` values which remain live
/// for the duration of this call.
unsafe fn planned_row_count(protocol: &GafimeLaunchProtocol) -> OrchestratorResult<u64> {
    let chunks = slice_from_parts(protocol.chunks, protocol.chunk_count as u64)?;
    Ok(chunks
        .iter()
        .fold(0u64, |total, chunk| total.saturating_add(chunk.combo_count)))
}

/// Borrow a C ABI pointer/length pair as a Rust slice.
///
/// # Safety
///
/// When `len` is nonzero, `ptr` must be non-null, properly aligned, and point
/// to `len` initialized `T` values in one allocation. That storage must remain
/// live and immutable for the returned lifetime.
unsafe fn slice_from_parts<'a, T>(ptr: *const T, len: u64) -> OrchestratorResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(OrchestratorError::InvalidPlan(
            "non-empty ABI slice has null pointer",
        ));
    }
    let len = usize::try_from(len).map_err(|_| {
        OrchestratorError::InvalidPlan("ABI slice length exceeds host address space")
    })?;
    Ok(core::slice::from_raw_parts(ptr, len))
}

/// Write one continuous result row using default rank metadata.
///
/// # Safety
///
/// Every result pointer must reference writable storage covering `output_row`
/// under the supplied `max_arity` and `metric_count` strides. `combo` and
/// `scores` must not exceed those respective strides.
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

/// Write one continuous result row and its explicit ranking metadata.
///
/// # Safety
///
/// Every result pointer must reference writable storage covering `output_row`
/// under the supplied `max_arity` and `metric_count` strides. `combo` and
/// `scores` must not exceed those respective strides.
#[allow(
    clippy::too_many_arguments,
    reason = "the ABI row writer keeps buffer strides, row identity, values, and ranking metadata explicit"
)]
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
    for index in 0..metric_count {
        *result.metric_values.add(metric_base + index) = scores.get(index).copied().unwrap_or(0.0);
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

// This hook is compiled only into Rust unit-test builds.  Production candidate
// loops retain no observer branch, allocation, synchronization, or global
// state.  The unique dedicated-pool name prevents unrelated parallel tests
// from being counted as precision-executor participation.
#[cfg(test)]
mod precision_parallelism_test_hook {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, MutexGuard, OnceLock,
    };

    #[derive(Default)]
    struct ParticipationState {
        pool_name_prefix: Option<String>,
        worker_mask: u64,
    }

    static PARTICIPATION: OnceLock<Mutex<ParticipationState>> = OnceLock::new();
    static RUN_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    static NEXT_RUN_ID: AtomicUsize = AtomicUsize::new(1);

    pub(super) struct ParticipationRun {
        pool_name_prefix: String,
        _run_lock: MutexGuard<'static, ()>,
    }

    pub(super) fn begin() -> ParticipationRun {
        let run_lock = RUN_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pool_name_prefix = format!(
            "gafime-precision-parallelism-test-{}",
            NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed)
        );
        let mut state = PARTICIPATION
            .get_or_init(|| Mutex::new(ParticipationState::default()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.pool_name_prefix = Some(pool_name_prefix.clone());
        state.worker_mask = 0;
        drop(state);
        ParticipationRun {
            pool_name_prefix,
            _run_lock: run_lock,
        }
    }

    impl ParticipationRun {
        pub(super) fn pool_name_prefix(&self) -> &str {
            &self.pool_name_prefix
        }

        pub(super) fn worker_count(&self) -> usize {
            PARTICIPATION
                .get_or_init(|| Mutex::new(ParticipationState::default()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .worker_mask
                .count_ones() as usize
        }
    }

    impl Drop for ParticipationRun {
        fn drop(&mut self) {
            let mut state = PARTICIPATION
                .get_or_init(|| Mutex::new(ParticipationState::default()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.pool_name_prefix = None;
            state.worker_mask = 0;
        }
    }

    pub(super) fn record_candidate_worker() {
        let current_thread = std::thread::current();
        let Some(thread_name) = current_thread.name() else {
            return;
        };
        let Some(worker_index) = rayon::current_thread_index() else {
            return;
        };
        if worker_index >= u64::BITS as usize {
            return;
        }
        let mut state = PARTICIPATION
            .get_or_init(|| Mutex::new(ParticipationState::default()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state
            .pool_name_prefix
            .as_deref()
            .is_some_and(|prefix| thread_name.starts_with(prefix))
        {
            state.worker_mask |= 1u64 << worker_index;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::CpuPrecisionValues;
    use gafime_orchestrator::{CompiledPlan, ComputeBackend, PrecisionComputeBackend};

    #[derive(Debug, PartialEq, Eq)]
    struct PrecisionExecutionSnapshot {
        row_count: u64,
        combo_indices: Vec<u32>,
        metric_bits: Vec<u64>,
        ranks: Vec<u32>,
        families: Vec<u32>,
        candidate_ids: Vec<u64>,
        row_flags: Vec<u32>,
    }

    const PARALLELISM_TEST_SAMPLES: usize = 1_024;
    const PARALLELISM_TEST_FEATURES: usize = 8;
    const PARALLELISM_TEST_CANDIDATES: usize = 1_024;
    const PARALLELISM_TEST_TOP_K: u32 = 64;
    const PARALLELISM_TEST_METRICS: u32 = 4;

    fn precision_parallelism_plan(ranked: bool) -> CompiledPlan {
        use gafime_types::{
            GafimeRankSpec, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_MUTUAL_INFO,
            GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
        };

        let combos = (0..PARALLELISM_TEST_CANDIDATES)
            .map(|candidate| (candidate % PARALLELISM_TEST_FEATURES) as u32)
            .collect();
        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            PARALLELISM_TEST_SAMPLES as u64,
            PARALLELISM_TEST_FEATURES as u32,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            combos,
            vec![
                GAFIME_METRIC_PEARSON,
                GAFIME_METRIC_SPEARMAN,
                GAFIME_METRIC_MUTUAL_INFO,
                GAFIME_METRIC_R2,
            ],
        );
        if ranked {
            plan.with_rank(GafimeRankSpec {
                top_k: PARALLELISM_TEST_TOP_K,
                primary_metric: GAFIME_METRIC_PEARSON,
                descending: 1,
                include_ties: 0,
                reserved: [0; 4],
            })
        } else {
            plan
        }
    }

    fn precision_parallelism_input_f32() -> (Vec<f32>, Vec<f32>) {
        let mut features = Vec::with_capacity(PARALLELISM_TEST_SAMPLES * PARALLELISM_TEST_FEATURES);
        let mut target = Vec::with_capacity(PARALLELISM_TEST_SAMPLES);
        for row in 0..PARALLELISM_TEST_SAMPLES {
            let row_value = row as f32;
            target.push(
                ((row * 17 % 251) as f32 * 0.03125)
                    + (row_value * 0.0078125).sin()
                    + ((row % 13) as f32 * 0.011),
            );
            for feature in 0..PARALLELISM_TEST_FEATURES {
                features.push(
                    ((row * (feature + 3) % 257) as f32 * 0.015625)
                        + (row_value * (feature as f32 + 1.0) * 0.00390625).cos()
                        + (feature as f32 * 0.021),
                );
            }
        }
        (features, target)
    }

    fn snapshot_f32(table: &result::OwnedResultTable) -> PrecisionExecutionSnapshot {
        let row_count = table.raw().row_count as usize;
        let max_arity = table.raw().max_arity as usize;
        let metric_count = table.raw().metric_count as usize;
        PrecisionExecutionSnapshot {
            row_count: row_count as u64,
            combo_indices: table.combo_indices()[..row_count * max_arity].to_vec(),
            metric_bits: table.metric_values()[..row_count * metric_count]
                .iter()
                .map(|value| u64::from(value.to_bits()))
                .collect(),
            ranks: table.ranks()[..row_count].to_vec(),
            families: table.families()[..row_count].to_vec(),
            candidate_ids: table.candidate_ids()[..row_count].to_vec(),
            row_flags: table.row_flags()[..row_count].to_vec(),
        }
    }

    fn snapshot_f64(table: &result::OwnedResultTableF64) -> PrecisionExecutionSnapshot {
        let row_count = table.raw().row_count as usize;
        let max_arity = table.raw().max_arity as usize;
        let metric_count = table.raw().metric_count as usize;
        PrecisionExecutionSnapshot {
            row_count: row_count as u64,
            combo_indices: table.combo_indices()[..row_count * max_arity].to_vec(),
            metric_bits: table.metric_values()[..row_count * metric_count]
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            ranks: table.ranks()[..row_count].to_vec(),
            families: table.families()[..row_count].to_vec(),
            candidate_ids: table.candidate_ids()[..row_count].to_vec(),
            row_flags: table.row_flags()[..row_count].to_vec(),
        }
    }

    fn fill_f32_metric_sentinel(table: &mut result::OwnedResultTable, sentinel: f32) {
        let raw = table.raw_mut();
        let len = raw.capacity as usize * raw.metric_count as usize;
        // SAFETY: OwnedResultTable::raw_mut rebinds this initialized buffer and
        // the ABI dimensions describe exactly its capacity.
        unsafe {
            std::slice::from_raw_parts_mut(raw.metric_values, len).fill(sentinel);
        }
    }

    fn fill_f64_metric_sentinel(table: &mut result::OwnedResultTableF64, sentinel: f64) {
        let raw = table.raw_mut();
        let len = raw.capacity as usize * raw.metric_count as usize;
        // SAFETY: OwnedResultTableF64::raw_mut rebinds this initialized buffer
        // and the ABI dimensions describe exactly its capacity.
        unsafe {
            std::slice::from_raw_parts_mut(raw.metric_values, len).fill(sentinel);
        }
    }

    fn execute_owned_precision_fp32(
        backend: &mut CpuBackend,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut result::OwnedResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        // SAFETY: callers supply a live materialized protocol and the owned
        // table rebinds all output pointers to its uniquely borrowed buffers.
        unsafe {
            PrecisionComputeBackend::execute_fp32(backend, matrix, protocol, result.raw_mut())
        }
    }

    fn execute_owned_precision_f64(
        backend: &mut CpuBackend,
        matrix: &MatrixHandle,
        protocol: &GafimePrecisionLaunchProtocol,
        result: &mut result::OwnedResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats> {
        // SAFETY: callers supply a live materialized protocol and the owned
        // table rebinds all output pointers to its uniquely borrowed buffers.
        unsafe { PrecisionComputeBackend::execute_f64(backend, matrix, protocol, result.raw_mut()) }
    }

    fn execute_owned_plan(
        backend: &mut CpuBackend,
        matrix: &MatrixHandle,
        plan: &CompiledPlan,
        result: &mut result::OwnedResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        // SAFETY: `plan` owns the complete protocol pointer graph and the
        // owned table rebinds outputs to correctly sized borrowed buffers.
        unsafe { gafime_orchestrator::execute_plan(backend, matrix, plan, result.raw_mut()) }
    }

    fn execute_precision_parallelism_case(
        profile: PrecisionProfile,
        ranked: bool,
    ) -> PrecisionExecutionSnapshot {
        use gafime_types::{GafimePrecisionLaunchProtocol, GAFIME_PRECISION_ABI_VERSION};

        let plan = precision_parallelism_plan(ranked);
        let base = plan.materialized_protocol();
        let protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: profile as u32,
            base: &base,
            reserved: [0; 8],
        };
        let (features_f32, target_f32) = precision_parallelism_input_f32();
        let mut backend = CpuBackend;
        match profile {
            PrecisionProfile::Fp32 => {
                let matrix = CpuPrecisionMatrix::from_row_major_f32(
                    profile,
                    PARALLELISM_TEST_SAMPLES as u64,
                    PARALLELISM_TEST_FEATURES as u32,
                    features_f32,
                    target_f32,
                )
                .unwrap();
                let handle = matrix.handle();
                let capacity = if ranked {
                    PARALLELISM_TEST_TOP_K as u64
                } else {
                    PARALLELISM_TEST_CANDIDATES as u64
                };
                let mut result =
                    result::OwnedResultTable::new(capacity, 1, PARALLELISM_TEST_METRICS);
                let stats =
                    execute_owned_precision_fp32(&mut backend, &handle, &protocol, &mut result)
                        .unwrap();
                assert_eq!(stats.rows_written, result.raw().row_count);
                assert!(result.metric_values()
                    [..result.raw().row_count as usize * PARALLELISM_TEST_METRICS as usize]
                    .iter()
                    .all(|value| value.is_finite()));
                snapshot_f32(&result)
            }
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
                let matrix = if profile == PrecisionProfile::Mixed {
                    CpuPrecisionMatrix::from_row_major_f32(
                        profile,
                        PARALLELISM_TEST_SAMPLES as u64,
                        PARALLELISM_TEST_FEATURES as u32,
                        features_f32,
                        target_f32,
                    )
                    .unwrap()
                } else {
                    CpuPrecisionMatrix::from_row_major_f64(
                        profile,
                        PARALLELISM_TEST_SAMPLES as u64,
                        PARALLELISM_TEST_FEATURES as u32,
                        features_f32.into_iter().map(f64::from).collect(),
                        target_f32.into_iter().map(f64::from).collect(),
                    )
                    .unwrap()
                };
                let handle = matrix.handle();
                let capacity = if ranked {
                    PARALLELISM_TEST_TOP_K as u64
                } else {
                    PARALLELISM_TEST_CANDIDATES as u64
                };
                let mut result =
                    result::OwnedResultTableF64::new(capacity, 1, PARALLELISM_TEST_METRICS);
                let stats =
                    execute_owned_precision_f64(&mut backend, &handle, &protocol, &mut result)
                        .unwrap();
                assert_eq!(stats.rows_written, result.raw().row_count);
                assert!(result.metric_values()
                    [..result.raw().row_count as usize * PARALLELISM_TEST_METRICS as usize]
                    .iter()
                    .all(|value| value.is_finite()));
                snapshot_f64(&result)
            }
        }
    }

    fn run_precision_parallelism_case(
        profile: PrecisionProfile,
        ranked: bool,
        worker_count: usize,
    ) -> (PrecisionExecutionSnapshot, usize) {
        let participation = precision_parallelism_test_hook::begin();
        let pool_name_prefix = participation.pool_name_prefix().to_owned();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(move |worker| format!("{pool_name_prefix}-{worker}"))
            .build()
            .unwrap();
        let snapshot = pool.install(|| execute_precision_parallelism_case(profile, ranked));
        let observed_workers = participation.worker_count();
        (snapshot, observed_workers)
    }

    #[test]
    fn cpu_backend_declares_cpu_kind() {
        assert_eq!(
            ComputeBackend::backend_kind(&CpuBackend),
            GAFIME_BACKEND_CPU
        );
        assert_eq!(
            PrecisionComputeBackend::backend_kind(&CpuBackend),
            GAFIME_BACKEND_CPU
        );
    }

    #[test]
    fn precision_executor_parallelism_contract_covers_every_profile_and_rank_mode() {
        let available_workers = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let multi_worker_count = available_workers.clamp(1, 4);

        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            for ranked in [false, true] {
                let (single_thread, single_thread_observed) =
                    run_precision_parallelism_case(profile, ranked, 1);
                assert!(
                    single_thread_observed <= 1,
                    "one-worker pool unexpectedly recorded {single_thread_observed} precision workers for {profile:?}, ranked={ranked}",
                );

                let (multi_thread, multi_thread_observed) =
                    run_precision_parallelism_case(profile, ranked, multi_worker_count);
                assert_eq!(
                    single_thread, multi_thread,
                    "Rayon scheduling changed Core precision results for {profile:?}, ranked={ranked}",
                );

                eprintln!(
                    "precision executor participation: profile={profile:?}, ranked={ranked}, pool={multi_worker_count}, observed={multi_thread_observed}"
                );

                if available_workers > 1 {
                    assert!(
                        multi_thread_observed > 1,
                        "Core precision executor did not engage multiple Rayon workers for {profile:?}, ranked={ranked}; pool={multi_worker_count}, observed={multi_thread_observed}",
                    );
                } else {
                    eprintln!(
                        "skipping multi-worker participation assertion for {profile:?}, ranked={ranked}: available_parallelism reports one usable processor"
                    );
                }
            }
        }
    }

    const ORACLE_SAMPLES: usize = 64;
    const ORACLE_FEATURES: usize = 5;
    const ORACLE_CANDIDATES: usize = 10;
    const ORACLE_TOP_K: u32 = 8;

    #[derive(Clone, Debug)]
    enum OraclePrimary {
        F32(f32),
        F64(f64),
    }

    #[derive(Clone, Debug)]
    struct OracleRow {
        candidate_id: u64,
        combo: Vec<u32>,
        metric_bits: Vec<u64>,
        primary: OraclePrimary,
    }

    fn precision_oracle_combos() -> Vec<Vec<u32>> {
        vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 4],
            vec![2, 4],
        ]
    }

    fn precision_oracle_plan(ranked: bool, descending: bool) -> CompiledPlan {
        use gafime_orchestrator::plan::shapes::default_shape_hint;
        use gafime_types::{
            GafimeArityChunk, GafimePermutationSchedule, GafimeRankSpec, GAFIME_FAMILY_CONTINUOUS,
            GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON,
            GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
        };

        let combos = precision_oracle_combos();
        let combo_indices = combos.iter().flatten().copied().collect::<Vec<_>>();
        let mut unary_shape = default_shape_hint(GAFIME_BACKEND_CPU, 1);
        unary_shape.vendor_hint = 8;
        let mut pair_shape = default_shape_hint(GAFIME_BACKEND_CPU, 2);
        pair_shape.vendor_hint = 8;
        CompiledPlan::from_parts(
            GAFIME_BACKEND_CPU,
            ORACLE_SAMPLES as u64,
            ORACLE_FEATURES as u32,
            2,
            combo_indices,
            vec![
                GAFIME_METRIC_PEARSON,
                GAFIME_METRIC_SPEARMAN,
                GAFIME_METRIC_MUTUAL_INFO,
                GAFIME_METRIC_R2,
            ],
            vec![
                GafimeArityChunk {
                    arity: 1,
                    family: GAFIME_FAMILY_CONTINUOUS,
                    shape_hint_index: 0,
                    combo_count: 5,
                    descriptor_count: 5,
                    ..Default::default()
                },
                GafimeArityChunk {
                    arity: 2,
                    family: GAFIME_FAMILY_CONTINUOUS,
                    shape_hint_index: 1,
                    combo_row_offset: 5,
                    combo_count: 5,
                    local_chunk_id: 1,
                    descriptor_offset: 5,
                    descriptor_count: 5,
                    ..Default::default()
                },
            ],
            vec![unary_shape, pair_shape],
            if ranked {
                GafimeRankSpec {
                    top_k: ORACLE_TOP_K,
                    primary_metric: GAFIME_METRIC_PEARSON,
                    descending: u32::from(descending),
                    include_ties: 0,
                    reserved: [0; 4],
                }
            } else {
                GafimeRankSpec::default()
            },
            GafimePermutationSchedule::default(),
        )
        .with_flags(GAFIME_LAUNCH_FLAG_MI_APPROX)
    }

    fn precision_oracle_matrix(profile: PrecisionProfile) -> CpuPrecisionMatrix {
        match profile {
            PrecisionProfile::Fp32 | PrecisionProfile::Mixed => {
                let mut features = Vec::with_capacity(ORACLE_SAMPLES * ORACLE_FEATURES);
                let mut target = Vec::with_capacity(ORACLE_SAMPLES);
                for row in 0..ORACLE_SAMPLES {
                    let target_value = row as f32 - 31.5;
                    target.push(target_value);
                    let huge = if row % 2 == 0 { f32::MAX } else { -f32::MAX };
                    let patterned = ((row * 19 + row / 3) % 23) as f32 - 11.0;
                    features.extend_from_slice(&[
                        target_value,
                        target_value,
                        -target_value,
                        huge,
                        patterned,
                    ]);
                }
                CpuPrecisionMatrix::from_row_major_f32(
                    profile,
                    ORACLE_SAMPLES as u64,
                    ORACLE_FEATURES as u32,
                    features,
                    target,
                )
                .unwrap()
            }
            PrecisionProfile::Fp64 => {
                let mut features = Vec::with_capacity(ORACLE_SAMPLES * ORACLE_FEATURES);
                let mut target = Vec::with_capacity(ORACLE_SAMPLES);
                let step = f64::EPSILON * 8.0;
                let mut fp64_only_column = Vec::with_capacity(ORACLE_SAMPLES);
                for row in 0..ORACLE_SAMPLES {
                    let target_value = row as f64 - 31.5;
                    let ascending = 1.0 + row as f64 * step;
                    fp64_only_column.push(ascending);
                    target.push(target_value);
                    let huge = if row % 2 == 0 { f64::MAX } else { -f64::MAX };
                    let patterned = ((row * 19 + row / 3) % 23) as f64 - 11.0;
                    features.extend_from_slice(&[
                        ascending,
                        ascending,
                        2.0 - row as f64 * step,
                        huge,
                        patterned,
                    ]);
                }
                assert!(
                    fp64_only_column
                        .iter()
                        .map(|&value| value as f32)
                        .all(|value| value == 1.0),
                    "fp64 oracle must contain distinctions that collapse in fp32"
                );
                assert!(fp64_only_column.windows(2).all(|pair| pair[0] != pair[1]));
                CpuPrecisionMatrix::from_row_major_f64(
                    profile,
                    ORACLE_SAMPLES as u64,
                    ORACLE_FEATURES as u32,
                    features,
                    target,
                )
                .unwrap()
            }
        }
    }

    fn precision_oracle_rows(
        profile: PrecisionProfile,
        matrix: &CpuPrecisionMatrix,
    ) -> Vec<OracleRow> {
        use kernels::precision::score_precision_continuous_combo;
        use kernels::MetricKernel;

        let metrics = [
            MetricKernel::Pearson,
            MetricKernel::Spearman,
            MetricKernel::MutualInfo,
            MetricKernel::R2,
        ];
        precision_oracle_combos()
            .into_iter()
            .enumerate()
            .map(|(candidate_index, combo)| {
                let candidate_id = candidate_index as u64;
                let scores =
                    score_precision_continuous_combo(matrix, &combo, &metrics, 8, true).unwrap();
                match (profile, scores) {
                    (PrecisionProfile::Fp32, CpuPrecisionValues::F32(scores)) => OracleRow {
                        candidate_id,
                        combo,
                        metric_bits: scores
                            .iter()
                            .map(|value| u64::from(value.to_bits()))
                            .collect(),
                        primary: OraclePrimary::F32(scores[0]),
                    },
                    (
                        PrecisionProfile::Mixed | PrecisionProfile::Fp64,
                        CpuPrecisionValues::F64(scores),
                    ) => OracleRow {
                        candidate_id,
                        combo,
                        metric_bits: scores.iter().map(|value| value.to_bits()).collect(),
                        primary: OraclePrimary::F64(scores[0]),
                    },
                    _ => panic!("profile scorer returned the wrong public result dtype"),
                }
            })
            .collect()
    }

    fn oracle_primary_is_finite(primary: &OraclePrimary) -> bool {
        match primary {
            OraclePrimary::F32(value) => value.is_finite(),
            OraclePrimary::F64(value) => value.is_finite(),
        }
    }

    fn oracle_primary_order(left: &OraclePrimary, right: &OraclePrimary) -> core::cmp::Ordering {
        match (left, right) {
            (OraclePrimary::F32(left), OraclePrimary::F32(right)) => left
                .partial_cmp(right)
                .unwrap_or(core::cmp::Ordering::Equal),
            (OraclePrimary::F64(left), OraclePrimary::F64(right)) => left
                .partial_cmp(right)
                .unwrap_or(core::cmp::Ordering::Equal),
            _ => panic!("an oracle comparison mixed public result dtypes"),
        }
    }

    fn precision_oracle_expected_snapshot(
        profile: PrecisionProfile,
        matrix: &CpuPrecisionMatrix,
        ranked: bool,
        descending: bool,
        result_max_arity: usize,
        result_metric_count: usize,
    ) -> PrecisionExecutionSnapshot {
        use gafime_types::GAFIME_FAMILY_CONTINUOUS;

        let mut rows = precision_oracle_rows(profile, matrix);
        if profile != PrecisionProfile::Mixed {
            assert!(
                !oracle_primary_is_finite(&rows[3].primary),
                "the fp32/fp64 oracle fixture must exercise ranked non-finite filtering, got {:?}",
                rows[3].primary
            );
        }
        assert_eq!(
            rows[0].metric_bits[0], rows[1].metric_bits[0],
            "the oracle fixture must contain an exact primary-score tie"
        );
        if ranked {
            rows.retain(|row| oracle_primary_is_finite(&row.primary));
            rows.sort_by(|left, right| {
                let ordering = oracle_primary_order(&left.primary, &right.primary);
                let ordering = if descending {
                    ordering.reverse()
                } else {
                    ordering
                };
                ordering.then(left.candidate_id.cmp(&right.candidate_id))
            });
            rows.truncate(ORACLE_TOP_K as usize);
        }

        let mut combo_indices = Vec::with_capacity(rows.len() * result_max_arity);
        let mut metric_bits = Vec::with_capacity(rows.len() * result_metric_count);
        for row in &rows {
            assert!(row.combo.len() <= result_max_arity);
            assert!(row.metric_bits.len() <= result_metric_count);
            combo_indices.extend_from_slice(&row.combo);
            combo_indices.resize(
                combo_indices.len() + (result_max_arity - row.combo.len()),
                u32::MAX,
            );
            metric_bits.extend_from_slice(&row.metric_bits);
            metric_bits.resize(
                metric_bits.len() + (result_metric_count - row.metric_bits.len()),
                0,
            );
        }
        PrecisionExecutionSnapshot {
            row_count: rows.len() as u64,
            combo_indices,
            metric_bits,
            ranks: (0..rows.len() as u32).collect(),
            families: vec![GAFIME_FAMILY_CONTINUOUS; rows.len()],
            candidate_ids: rows.iter().map(|row| row.candidate_id).collect(),
            row_flags: vec![0; rows.len()],
        }
    }

    fn execute_precision_oracle_case(
        profile: PrecisionProfile,
        ranked: bool,
        descending: bool,
        result_max_arity: usize,
        result_metric_count: usize,
    ) -> PrecisionExecutionSnapshot {
        use gafime_types::{GafimePrecisionLaunchProtocol, GAFIME_PRECISION_ABI_VERSION};

        let plan = precision_oracle_plan(ranked, descending);
        let base = plan.materialized_protocol();
        let protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: profile as u32,
            base: &base,
            reserved: [0; 8],
        };
        let matrix = precision_oracle_matrix(profile);
        let expected = precision_oracle_expected_snapshot(
            profile,
            &matrix,
            ranked,
            descending,
            result_max_arity,
            result_metric_count,
        );
        let handle = matrix.handle();
        let capacity = if ranked {
            ORACLE_TOP_K as u64
        } else {
            ORACLE_CANDIDATES as u64
        };
        let mut backend = CpuBackend;
        let actual = match profile {
            PrecisionProfile::Fp32 => {
                let mut result = result::OwnedResultTable::new(
                    capacity,
                    result_max_arity as u32,
                    result_metric_count as u32,
                );
                fill_f32_metric_sentinel(&mut result, f32::MAX);
                execute_owned_precision_fp32(&mut backend, &handle, &protocol, &mut result)
                    .unwrap();
                snapshot_f32(&result)
            }
            PrecisionProfile::Mixed | PrecisionProfile::Fp64 => {
                let mut result = result::OwnedResultTableF64::new(
                    capacity,
                    result_max_arity as u32,
                    result_metric_count as u32,
                );
                fill_f64_metric_sentinel(&mut result, f64::MAX);
                execute_owned_precision_f64(&mut backend, &handle, &protocol, &mut result).unwrap();
                snapshot_f64(&result)
            }
        };
        assert_eq!(actual, expected);
        actual
    }

    #[test]
    fn precision_executor_matches_independent_multichunk_ranking_oracle() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            let unranked =
                pool.install(|| execute_precision_oracle_case(profile, false, false, 2, 4));
            assert_eq!(
                unranked.candidate_ids,
                (0..ORACLE_CANDIDATES as u64).collect::<Vec<_>>()
            );
            assert!(match profile {
                PrecisionProfile::Fp32 => {
                    f32::from_bits(unranked.metric_bits[3 * 4] as u32).is_nan()
                }
                PrecisionProfile::Fp64 => {
                    f64::from_bits(unranked.metric_bits[3 * 4]).is_nan()
                }
                PrecisionProfile::Mixed => {
                    f64::from_bits(unranked.metric_bits[3 * 4]).is_finite()
                }
            });

            for descending in [false, true] {
                let ranked =
                    pool.install(|| execute_precision_oracle_case(profile, true, descending, 2, 4));
                assert_eq!(ranked.row_count, ORACLE_TOP_K as u64);
                if profile != PrecisionProfile::Mixed {
                    assert!(!ranked.candidate_ids.contains(&3));
                }
                if descending {
                    let first_tie = ranked
                        .candidate_ids
                        .iter()
                        .position(|&candidate| candidate == 0)
                        .unwrap();
                    let second_tie = ranked
                        .candidate_ids
                        .iter()
                        .position(|&candidate| candidate == 1)
                        .unwrap();
                    assert!(first_tie < second_tie);
                }
            }
        }
    }

    #[test]
    fn precision_executor_uses_wider_result_stride_and_pads_combo_rows() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            for ranked in [false, true] {
                let descending_modes = if ranked {
                    [false, true]
                } else {
                    [false, false]
                };
                for descending in descending_modes {
                    let snapshot = pool.install(|| {
                        execute_precision_oracle_case(profile, ranked, descending, 3, 4)
                    });
                    assert_eq!(
                        snapshot.combo_indices.len(),
                        snapshot.row_count as usize * 3
                    );
                    for row in snapshot.combo_indices.chunks_exact(3) {
                        assert_eq!(row[2], u32::MAX);
                    }
                }
            }
        }
    }

    #[test]
    fn precision_executor_zero_fills_wider_metric_stride_for_rank_modes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for profile in [
            PrecisionProfile::Fp32,
            PrecisionProfile::Mixed,
            PrecisionProfile::Fp64,
        ] {
            for ranked in [false, true] {
                let snapshot =
                    pool.install(|| execute_precision_oracle_case(profile, ranked, ranked, 2, 6));
                for row in snapshot.metric_bits.chunks_exact(6) {
                    assert_eq!(&row[4..], &[0, 0]);
                }
            }
        }
    }

    #[test]
    fn precision_backend_executes_all_three_profiles_with_typed_result_tables() {
        use gafime_types::{
            GafimePrecisionLaunchProtocol, GAFIME_FAMILY_CONTINUOUS, GAFIME_METRIC_PEARSON,
            GAFIME_METRIC_R2, GAFIME_PRECISION_ABI_VERSION,
        };

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_CPU,
            4,
            1,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0],
            vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
        );
        let base = plan.materialized_protocol();
        let mut backend = CpuBackend;

        let fp32 = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Fp32,
            4,
            1,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
        )
        .unwrap();
        let fp32_handle = fp32.handle();
        let fp32_protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: PrecisionProfile::Fp32 as u32,
            base: &base,
            reserved: [0; 8],
        };
        let mut fp32_result = result::OwnedResultTable::new(1, 1, 2);
        execute_owned_precision_fp32(&mut backend, &fp32_handle, &fp32_protocol, &mut fp32_result)
            .unwrap();
        assert_eq!(fp32_result.raw().row_count, 1);
        assert_eq!(fp32_result.metric_values()[0], 1.0);

        let mixed = CpuPrecisionMatrix::from_row_major_f32(
            PrecisionProfile::Mixed,
            4,
            1,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
        )
        .unwrap();
        let mixed_handle = mixed.handle();
        let mixed_protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: PrecisionProfile::Mixed as u32,
            base: &base,
            reserved: [0; 8],
        };
        let mut mixed_result = result::OwnedResultTableF64::new(1, 1, 2);
        execute_owned_precision_f64(
            &mut backend,
            &mixed_handle,
            &mixed_protocol,
            &mut mixed_result,
        )
        .unwrap();
        assert_eq!(mixed_result.raw().row_count, 1);
        assert_eq!(mixed_result.metric_values()[0], 1.0);

        let base_value = 1.0f64;
        let fp64 = CpuPrecisionMatrix::from_row_major_f64(
            PrecisionProfile::Fp64,
            4,
            1,
            vec![
                base_value,
                f64::from_bits(base_value.to_bits() + 1024),
                f64::from_bits(base_value.to_bits() + 2048),
                f64::from_bits(base_value.to_bits() + 3072),
            ],
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let fp64_handle = fp64.handle();
        let fp64_protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: PrecisionProfile::Fp64 as u32,
            base: &base,
            reserved: [0; 8],
        };
        let mut fp64_result = result::OwnedResultTableF64::new(1, 1, 2);
        execute_owned_precision_f64(&mut backend, &fp64_handle, &fp64_protocol, &mut fp64_result)
            .unwrap();
        assert_eq!(fp64_result.raw().row_count, 1);
        assert!((fp64_result.metric_values()[0] - 1.0).abs() < 1.0e-12);
        assert_eq!(fp64_result.candidate_ids()[0], 0);
    }

    #[test]
    fn cpu_backend_executes_continuous_result_table() {
        use gafime_orchestrator::CompiledPlan;
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
        let mut table = result::OwnedResultTable::new(3, 2, 3);
        fill_f32_metric_sentinel(&mut table, f32::MAX);
        let mut backend = CpuBackend;

        let stats = execute_owned_plan(&mut backend, &handle, &plan, &mut table).unwrap();

        assert_eq!(stats.rows_written, 3);
        assert_eq!(table.raw().row_count, 3);
        assert_eq!(
            &table.combo_indices()[..6],
            &[0, u32::MAX, 1, u32::MAX, 2, u32::MAX]
        );
        for padding in [2, 5, 8] {
            assert_eq!(table.metric_values()[padding], 0.0);
        }
        assert!((table.metric_values()[0] - 1.0).abs() < 1e-6);
        assert!((table.metric_values()[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_backend_executes_every_adaptive_fixed_mi_template() {
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
                precision: PrecisionProfile::Fp32,
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

            execute_owned_plan(&mut backend, &matrix.handle(), &plan, &mut table).unwrap();

            let expected = kernels::mutual_info_fixed(&feature, &target, bins);
            assert_eq!(table.metric_values()[0], expected, "bins={bins}");
        }
    }

    #[test]
    fn cpu_backend_honors_rank_top_k() {
        use gafime_orchestrator::CompiledPlan;
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
        let mut table = result::OwnedResultTable::new(2, 2, 3);
        fill_f32_metric_sentinel(&mut table, f32::MAX);
        let mut backend = CpuBackend;

        let stats = execute_owned_plan(&mut backend, &matrix.handle(), &plan, &mut table).unwrap();

        assert_eq!(stats.rows_written, 2);
        assert_eq!(table.raw().row_count, 2);
        assert_eq!(&table.combo_indices()[..4], &[0, u32::MAX, 1, u32::MAX]);
        assert_eq!(table.metric_values()[2], 0.0);
        assert_eq!(table.metric_values()[5], 0.0);
        assert_eq!(table.metric_values()[1], 1.0);
        assert_eq!(table.metric_values()[4], 1.0);
        assert_eq!(&table.candidate_ids()[..2], &[0, 1]);
    }

    #[test]
    fn cpu_backend_ranks_arity_two_scores_like_materialized_reference() {
        use gafime_orchestrator::CompiledPlan;
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

        let stats = execute_owned_plan(&mut backend, &matrix.handle(), &plan, &mut table).unwrap();

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
            precision: PrecisionProfile::Fp32,
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

        let stats = execute_owned_plan(&mut backend, &matrix.handle(), &plan, &mut table).unwrap();

        assert_eq!(stats.rows_written, 6);
        assert_eq!(table.raw().row_count, 6);
        assert_eq!(
            &table.combo_indices()[..8],
            &[0, u32::MAX, 1, u32::MAX, 2, u32::MAX, 0, 1]
        );
    }
}
