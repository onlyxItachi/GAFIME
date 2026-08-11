use gafime_types::{
    BackendKind, GafimeLaunchProtocol, GafimePermutationSchedule, GafimePrecisionLaunchProtocol,
    GafimeRankSpec, GafimeResultTable, GafimeResultTableF64, GafimeSliceU32, PrecisionProfile,
    GAFIME_ABI_VERSION, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL,
    GAFIME_BACKEND_ROCM, GAFIME_LAUNCH_FLAG_GRAPH, GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_PRECISION_ABI_VERSION, GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    backend::{BackendExecutionStats, ComputeBackend, MatrixHandle, PrecisionComputeBackend},
    config::EngineConfig,
    plan::{
        combos::{
            build_continuous_plan, build_continuous_plan_for_feature_orders, ContinuousPlanRequest,
            DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES,
        },
        CompiledPlan, DEFAULT_DESCRIPTOR_BATCH_WORDS,
    },
    schedule::ContinuousSchedule,
    OrchestratorError, OrchestratorResult,
};

#[derive(Debug)]
pub struct PreparedContinuousExecution {
    plan: CompiledPlan,
    schedule: ContinuousSchedule,
    precision: PrecisionProfile,
    descriptor_generation: u64,
    rows: u64,
    cols: u32,
    device_budget_bytes: Option<u64>,
}

const DESCRIPTOR_GENERATION_RESERVED_SLOT: usize = 0;
const MAX_COMPATIBILITY_PROTOCOL_DESCRIPTOR_WORDS: u64 =
    DEFAULT_UNRANKED_HOST_STORAGE_BUDGET_BYTES / core::mem::size_of::<u32>() as u64;
static NEXT_DESCRIPTOR_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_descriptor_generation() -> u64 {
    loop {
        let generation = NEXT_DESCRIPTOR_GENERATION.fetch_add(1, Ordering::Relaxed);
        if generation != 0 {
            return generation;
        }
    }
}

impl PreparedContinuousExecution {
    pub fn plan(&self) -> &CompiledPlan {
        &self.plan
    }

    pub fn into_plan(self) -> CompiledPlan {
        self.plan
    }

    pub fn schedule(&self) -> ContinuousSchedule {
        self.schedule
    }

    pub fn precision(&self) -> PrecisionProfile {
        self.precision
    }

    pub fn result_capacity(&self) -> u64 {
        self.schedule.result_table().capacity()
    }

    pub fn result_max_arity(&self) -> u32 {
        self.schedule.result_table().max_arity()
    }

    pub fn result_metric_count(&self) -> u32 {
        self.schedule.result_table().metric_count()
    }

    /// Return a monolithic immutable protocol, materializing generated
    /// descriptors only within the compatibility host-memory bound.
    pub fn try_launch_protocol(&self) -> OrchestratorResult<GafimeLaunchProtocol> {
        let mut protocol = self
            .plan
            .try_materialized_protocol(MAX_COMPATIBILITY_PROTOCOL_DESCRIPTOR_WORDS)?;
        protocol.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT] = self.descriptor_generation;
        Ok(protocol)
    }

    /// Compatibility accessor for existing direct-ABI consumers. It preserves
    /// the infallible signature for bounded plans and fails explicitly instead
    /// of returning a protocol that cannot pass backend validation.
    pub fn launch_protocol(&self) -> GafimeLaunchProtocol {
        self.try_launch_protocol().expect(
            "launch_protocol compatibility materialization exceeds the bounded host-memory budget; use try_launch_protocol or execute",
        )
    }

    pub fn ranked_result_capacity(&self, rank: GafimeRankSpec) -> OrchestratorResult<u64> {
        validate_rank_override(&self.plan, rank)?;
        self.validate_rank_device_budget(rank)?;
        Ok(effective_ranked_rows(&self.plan, rank))
    }

    /// Execute a plan that was validated when this prepared artifact was built.
    /// General callers should use `execute_plan`, which validates arbitrary
    /// plans on every call; compiled artifacts keep this immutable trusted path.
    /// GPU adapters remove the hint unless the loaded payload advertises it.
    /// Execute the prepared plan into a raw ABI 1.0 result table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`ComputeBackend::execute`] for `matrix` and
    /// every pointer and declared extent in `result`.
    pub unsafe fn execute<B: ComputeBackend>(
        &self,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        execute_compiled_plan_with_device_budget(
            &self.plan,
            Some(self.descriptor_generation),
            self.device_budget_bytes,
            backend,
            matrix,
            result,
        )
    }

    /// Execute the prepared fp32 plan into a raw ABI 1.1 result table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`PrecisionComputeBackend::execute_fp32`] for
    /// `matrix` and every pointer and declared extent in `result`.
    pub unsafe fn execute_precision_fp32<B: PrecisionComputeBackend>(
        &self,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if self.precision != PrecisionProfile::Fp32 || matrix.precision() != self.precision {
            return Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity",
            ));
        }
        let mut adapter = PrecisionFp32BackendAdapter { backend };
        execute_compiled_plan_with_device_budget(
            &self.plan,
            Some(self.descriptor_generation),
            self.device_budget_bytes,
            &mut adapter,
            matrix,
            result,
        )
    }

    /// Execute the prepared mixed/fp64 plan into a raw ABI 1.1 result table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`PrecisionComputeBackend::execute_f64`] for
    /// `matrix` and every pointer and declared extent in `result`.
    pub unsafe fn execute_precision_f64<B: PrecisionComputeBackend>(
        &self,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if self.precision == PrecisionProfile::Fp32 || matrix.precision() != self.precision {
            return Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity",
            ));
        }
        execute_compiled_plan_f64_with_protocol(
            &self.plan,
            Some(self.descriptor_generation),
            self.plan.rank(),
            self.plan.permutations(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS,
            self.device_budget_bytes,
            backend,
            matrix,
            result,
        )
    }

    /// Score the complete prepared family with a bounded rank override. This is
    /// the generated-plan path for host-orchestrated maxT extrema: descriptors
    /// are streamed, only K rows are retained, and the plan's permutation
    /// schedule is disabled because the caller supplies the target being scored.
    /// Execute a ranked prepared plan into a raw ABI 1.0 result table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`ComputeBackend::execute`] for `matrix` and
    /// every pointer and declared extent in `result`.
    pub unsafe fn execute_ranked<B: ComputeBackend>(
        &self,
        rank: GafimeRankSpec,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        validate_rank_override(&self.plan, rank)?;
        self.validate_rank_device_budget(rank)?;
        execute_compiled_plan_with_protocol(
            &self.plan,
            Some(self.descriptor_generation),
            rank,
            GafimePermutationSchedule::default(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS,
            self.device_budget_bytes,
            backend,
            matrix,
            result,
        )
    }

    /// Execute a ranked prepared fp32 plan into a raw ABI 1.1 result table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`PrecisionComputeBackend::execute_fp32`] for
    /// `matrix` and every pointer and declared extent in `result`.
    pub unsafe fn execute_precision_ranked_fp32<B: PrecisionComputeBackend>(
        &self,
        rank: GafimeRankSpec,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if self.precision != PrecisionProfile::Fp32 || matrix.precision() != self.precision {
            return Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity",
            ));
        }
        validate_rank_override(&self.plan, rank)?;
        self.validate_rank_device_budget(rank)?;
        let mut adapter = PrecisionFp32BackendAdapter { backend };
        execute_compiled_plan_with_protocol(
            &self.plan,
            Some(self.descriptor_generation),
            rank,
            GafimePermutationSchedule::default(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS,
            self.device_budget_bytes,
            &mut adapter,
            matrix,
            result,
        )
    }

    /// Execute a ranked prepared mixed/fp64 plan into a raw ABI 1.1 table.
    ///
    /// # Safety
    ///
    /// The caller must satisfy [`PrecisionComputeBackend::execute_f64`] for
    /// `matrix` and every pointer and declared extent in `result`.
    pub unsafe fn execute_precision_ranked_f64<B: PrecisionComputeBackend>(
        &self,
        rank: GafimeRankSpec,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTableF64,
    ) -> OrchestratorResult<BackendExecutionStats> {
        if self.precision == PrecisionProfile::Fp32 || matrix.precision() != self.precision {
            return Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity",
            ));
        }
        validate_rank_override(&self.plan, rank)?;
        self.validate_rank_device_budget(rank)?;
        execute_compiled_plan_f64_with_protocol(
            &self.plan,
            Some(self.descriptor_generation),
            rank,
            GafimePermutationSchedule::default(),
            DEFAULT_DESCRIPTOR_BATCH_WORDS,
            self.device_budget_bytes,
            backend,
            matrix,
            result,
        )
    }

    fn validate_rank_device_budget(&self, rank: GafimeRankSpec) -> OrchestratorResult<()> {
        let Some(budget_bytes) = self.device_budget_bytes else {
            return Ok(());
        };
        let footprint = continuous_plan_device_footprint_bytes_for_rank(
            self.rows,
            self.cols,
            &self.plan,
            self.precision,
            rank,
        );
        if footprint > budget_bytes {
            return Err(OrchestratorError::Unsupported(
                "rank override device footprint exceeds budget.vram_budget_mb",
            ));
        }
        Ok(())
    }
}

struct PrecisionFp32BackendAdapter<'a, B> {
    backend: &'a mut B,
}

impl<B: PrecisionComputeBackend> ComputeBackend for PrecisionFp32BackendAdapter<'_, B> {
    fn backend_kind(&self) -> BackendKind {
        PrecisionComputeBackend::backend_kind(self.backend)
    }

    unsafe fn execution_device_memory_peak_bytes(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
    ) -> OrchestratorResult<Option<u64>> {
        let precision_protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: PrecisionProfile::Fp32 as u32,
            base: protocol,
            reserved: [0; 8],
        };
        // SAFETY: this adapter preserves the caller's raw protocol lifetime and
        // only adds a stack-local precision wrapper for the synchronous call.
        unsafe {
            self.backend
                .execution_device_memory_peak_bytes_v2(matrix, &precision_protocol)
        }
    }

    unsafe fn execute(
        &mut self,
        matrix: &MatrixHandle,
        protocol: &GafimeLaunchProtocol,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        let precision_protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: PrecisionProfile::Fp32 as u32,
            base: protocol,
            reserved: [0; 8],
        };
        // SAFETY: the adapter forwards the caller-provided result storage and
        // the same live base protocol through a stack-local precision wrapper.
        unsafe {
            self.backend
                .execute_fp32(matrix, &precision_protocol, result)
        }
    }
}

pub(crate) fn execute_compiled_plan<B: ComputeBackend>(
    plan: &CompiledPlan,
    descriptor_generation: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    execute_compiled_plan_with_device_budget(
        plan,
        descriptor_generation,
        None,
        backend,
        matrix,
        result,
    )
}

fn execute_compiled_plan_with_device_budget<B: ComputeBackend>(
    plan: &CompiledPlan,
    descriptor_generation: Option<u64>,
    device_budget_bytes: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    execute_compiled_plan_with_protocol(
        plan,
        descriptor_generation,
        plan.rank(),
        plan.permutations(),
        DEFAULT_DESCRIPTOR_BATCH_WORDS,
        device_budget_bytes,
        backend,
        matrix,
        result,
    )
}

#[cfg(test)]
fn execute_compiled_plan_with_batch_words<B: ComputeBackend>(
    plan: &CompiledPlan,
    descriptor_generation: Option<u64>,
    max_descriptor_words: usize,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    execute_compiled_plan_with_protocol(
        plan,
        descriptor_generation,
        plan.rank(),
        plan.permutations(),
        max_descriptor_words,
        None,
        backend,
        matrix,
        result,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_compiled_plan_with_protocol<B: ComputeBackend>(
    plan: &CompiledPlan,
    descriptor_generation: Option<u64>,
    rank: GafimeRankSpec,
    permutations: GafimePermutationSchedule,
    max_descriptor_words: usize,
    device_budget_bytes: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    if !plan.uses_generated_descriptors() {
        let mut protocol = plan.materialized_protocol();
        protocol.rank = rank;
        protocol.permutations = permutations;
        if let Some(generation) = descriptor_generation {
            protocol.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
            protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT] = generation;
        }
        validate_execution_device_budget(backend, matrix, &protocol, device_budget_bytes)?;
        // SAFETY: `protocol` is rebound exclusively to storage owned by `plan`
        // for this synchronous call. The public raw execution entry point makes
        // the caller responsible for the result-table allocation contract.
        return unsafe { backend.execute(matrix, &protocol, result) };
    }

    if rank.top_k == 0 {
        return Err(OrchestratorError::Unsupported(
            "generated continuous execution requires an explicit non-zero rank.top_k",
        ));
    }
    validate_generated_ranked_execution(plan, rank)?;
    execute_streamed_ranked_plan(
        plan,
        rank,
        permutations,
        max_descriptor_words,
        device_budget_bytes,
        backend,
        matrix,
        result,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the executor keeps independent plan, schedule, budget, backend, matrix, and ABI output state explicit"
)]
fn execute_streamed_ranked_plan<B: ComputeBackend>(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
    permutations: GafimePermutationSchedule,
    max_descriptor_words: usize,
    device_budget_bytes: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTable,
) -> OrchestratorResult<BackendExecutionStats> {
    // Under the backend's total order, every global top-K row must appear in its
    // own batch's top K. Merge only those bounded rows using the same finite-score,
    // direction, and candidate-id tie-break contract as the native selectors.
    let effective_top_k = effective_ranked_rows(plan, rank);
    let top_k = usize::try_from(effective_top_k).map_err(|_| {
        OrchestratorError::InvalidPlan("effective top-k exceeds the host address space")
    })?;
    let primary_metric_index = plan
        .metric_ids()
        .iter()
        .position(|&metric| metric == rank.primary_metric)
        .ok_or(OrchestratorError::InvalidPlan(
            "rank primary metric is not in the plan metric set",
        ))?;
    validate_ranked_result_table(plan, rank, result)?;

    let mut selected = Vec::with_capacity(top_k);
    let mut aggregate = BackendExecutionStats::default();
    let mut result_metadata = StreamedResultMetadata::new(result);
    result.row_count = 0;

    for batch in plan.descriptor_batches_validated(max_descriptor_words)? {
        let launch_chunk = batch.launch_chunk();
        let mut protocol = plan.protocol_template();
        protocol.combo_indices = GafimeSliceU32 {
            ptr: batch.combo_indices().as_ptr(),
            len: batch.combo_indices().len() as u64,
        };
        protocol.chunks = &launch_chunk;
        protocol.chunk_count = 1;
        let batch_capacity = batch.combo_count().min(effective_top_k);
        let mut batch_rank = rank;
        batch_rank.top_k = u32::try_from(batch_capacity).map_err(|_| {
            OrchestratorError::InvalidPlan("streamed batch top-k exceeds the launch ABI")
        })?;
        protocol.rank = batch_rank;
        protocol.permutations = permutations;
        protocol.flags &= !GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT] = 0;

        let mut batch_result =
            OwnedBatchResultTable::new(batch_capacity, plan.max_arity(), plan.metric_count())?;
        result_metadata.bind_batch(&mut batch_result.raw);
        validate_execution_device_budget(backend, matrix, &protocol, device_budget_bytes)?;
        // SAFETY: the batch owns every descriptor and output allocation, and
        // both remain live and uniquely borrowed for this synchronous call.
        let stats = unsafe { backend.execute(matrix, &protocol, &mut batch_result.raw) }?;
        if batch_result.raw.row_count > batch_capacity
            || stats.rows_written != batch_result.raw.row_count
        {
            return Err(OrchestratorError::InvalidPlan(
                "backend exceeded the streamed top-k batch capacity",
            ));
        }
        result_metadata.observe_batch(&batch_result.raw)?;
        aggregate.launched_chunks = aggregate
            .launched_chunks
            .saturating_add(stats.launched_chunks);
        aggregate.graph_replays = aggregate.graph_replays.saturating_add(stats.graph_replays);

        for row in 0..batch_result.raw.row_count as usize {
            let local_candidate_id = batch_result.candidate_ids[row];
            if local_candidate_id >= batch.combo_count() {
                return Err(OrchestratorError::InvalidPlan(
                    "backend returned a candidate outside its streamed batch",
                ));
            }
            let candidate_id = batch
                .logical_row_offset()
                .checked_add(local_candidate_id)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed candidate id overflows",
                ))?;
            let combo_base = row * plan.max_arity() as usize;
            let metric_base = row * plan.metric_count() as usize;
            let metrics = batch_result.metric_values
                [metric_base..metric_base + plan.metric_count() as usize]
                .to_vec();
            let score = metrics[primary_metric_index];
            if !score.is_finite() {
                continue;
            }
            consider_ranked_row(
                &mut selected,
                StreamedRankedRow {
                    score,
                    candidate_id,
                    combo: batch_result.combo_indices
                        [combo_base..combo_base + plan.max_arity() as usize]
                        .to_vec(),
                    metrics,
                    family: batch_result.families[row],
                    row_flags: batch_result.row_flags[row],
                },
                top_k,
                rank.descending != 0,
            );
        }
    }

    write_ranked_rows(result, &selected)?;
    result_metadata.finish(result);
    aggregate.rows_written = selected.len() as u64;
    Ok(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn execute_compiled_plan_f64_with_protocol<B: PrecisionComputeBackend>(
    plan: &CompiledPlan,
    descriptor_generation: Option<u64>,
    rank: GafimeRankSpec,
    permutations: GafimePermutationSchedule,
    max_descriptor_words: usize,
    device_budget_bytes: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTableF64,
) -> OrchestratorResult<BackendExecutionStats> {
    if !plan.uses_generated_descriptors() {
        let mut base = plan.materialized_protocol();
        base.rank = rank;
        base.permutations = permutations;
        if let Some(generation) = descriptor_generation {
            base.flags |= GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
            base.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT] = generation;
        }
        let protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: matrix.precision() as u32,
            base: &base,
            reserved: [0; 8],
        };
        validate_precision_execution_device_budget(
            backend,
            matrix,
            &protocol,
            device_budget_bytes,
        )?;
        // SAFETY: `base` and the precision wrapper are stack-local views over
        // live plan storage. The public raw execution entry point establishes
        // the caller's f64 result-table allocation contract.
        return unsafe { backend.execute_f64(matrix, &protocol, result) };
    }

    if rank.top_k == 0 {
        return Err(OrchestratorError::Unsupported(
            "generated continuous execution requires an explicit non-zero rank.top_k",
        ));
    }
    validate_generated_ranked_execution(plan, rank)?;
    execute_streamed_ranked_plan_f64(
        plan,
        rank,
        permutations,
        max_descriptor_words,
        device_budget_bytes,
        backend,
        matrix,
        result,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "typed streamed execution keeps the plan, budget, backend, matrix, and f64 ABI output explicit"
)]
fn execute_streamed_ranked_plan_f64<B: PrecisionComputeBackend>(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
    permutations: GafimePermutationSchedule,
    max_descriptor_words: usize,
    device_budget_bytes: Option<u64>,
    backend: &mut B,
    matrix: &MatrixHandle,
    result: &mut GafimeResultTableF64,
) -> OrchestratorResult<BackendExecutionStats> {
    let effective_top_k = effective_ranked_rows(plan, rank);
    let top_k = usize::try_from(effective_top_k).map_err(|_| {
        OrchestratorError::InvalidPlan("effective top-k exceeds the host address space")
    })?;
    let primary_metric_index = plan
        .metric_ids()
        .iter()
        .position(|&metric| metric == rank.primary_metric)
        .ok_or(OrchestratorError::InvalidPlan(
            "rank primary metric is not in the plan metric set",
        ))?;
    validate_ranked_result_table_f64(plan, rank, result)?;

    let mut selected = Vec::with_capacity(top_k);
    let mut aggregate = BackendExecutionStats::default();
    let mut result_metadata = StreamedResultMetadataF64::new(result);
    result.row_count = 0;

    for batch in plan.descriptor_batches_validated(max_descriptor_words)? {
        let launch_chunk = batch.launch_chunk();
        let mut base = plan.protocol_template();
        base.combo_indices = GafimeSliceU32 {
            ptr: batch.combo_indices().as_ptr(),
            len: batch.combo_indices().len() as u64,
        };
        base.chunks = &launch_chunk;
        base.chunk_count = 1;
        let batch_capacity = batch.combo_count().min(effective_top_k);
        let mut batch_rank = rank;
        batch_rank.top_k = u32::try_from(batch_capacity).map_err(|_| {
            OrchestratorError::InvalidPlan("streamed batch top-k exceeds the launch ABI")
        })?;
        base.rank = batch_rank;
        base.permutations = permutations;
        base.flags &= !GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
        base.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT] = 0;
        let protocol = GafimePrecisionLaunchProtocol {
            abi_version: GAFIME_PRECISION_ABI_VERSION,
            profile: matrix.precision() as u32,
            base: &base,
            reserved: [0; 8],
        };

        let mut batch_result =
            OwnedBatchResultTableF64::new(batch_capacity, plan.max_arity(), plan.metric_count())?;
        result_metadata.bind_batch(&mut batch_result.raw);
        validate_precision_execution_device_budget(
            backend,
            matrix,
            &protocol,
            device_budget_bytes,
        )?;
        // SAFETY: the batch owns every descriptor and typed output allocation,
        // and both remain live and uniquely borrowed for this call.
        let stats = unsafe { backend.execute_f64(matrix, &protocol, &mut batch_result.raw) }?;
        if batch_result.raw.row_count > batch_capacity
            || stats.rows_written != batch_result.raw.row_count
        {
            return Err(OrchestratorError::InvalidPlan(
                "backend exceeded the streamed top-k batch capacity",
            ));
        }
        result_metadata.observe_batch(&batch_result.raw)?;
        aggregate.launched_chunks = aggregate
            .launched_chunks
            .saturating_add(stats.launched_chunks);
        aggregate.graph_replays = aggregate.graph_replays.saturating_add(stats.graph_replays);

        for row in 0..batch_result.raw.row_count as usize {
            let local_candidate_id = batch_result.candidate_ids[row];
            if local_candidate_id >= batch.combo_count() {
                return Err(OrchestratorError::InvalidPlan(
                    "backend returned a candidate outside its streamed batch",
                ));
            }
            let candidate_id = batch
                .logical_row_offset()
                .checked_add(local_candidate_id)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed candidate id overflows",
                ))?;
            let combo_base = row * plan.max_arity() as usize;
            let metric_base = row * plan.metric_count() as usize;
            let metrics = batch_result.metric_values
                [metric_base..metric_base + plan.metric_count() as usize]
                .to_vec();
            let score = metrics[primary_metric_index];
            if !score.is_finite() {
                continue;
            }
            consider_ranked_row_f64(
                &mut selected,
                StreamedRankedRowF64 {
                    score,
                    candidate_id,
                    combo: batch_result.combo_indices
                        [combo_base..combo_base + plan.max_arity() as usize]
                        .to_vec(),
                    metrics,
                    family: batch_result.families[row],
                    row_flags: batch_result.row_flags[row],
                },
                top_k,
                rank.descending != 0,
            );
        }
    }

    write_ranked_rows_f64(result, &selected)?;
    result_metadata.finish(result);
    aggregate.rows_written = selected.len() as u64;
    Ok(aggregate)
}

fn validate_precision_execution_device_budget<B: PrecisionComputeBackend>(
    backend: &mut B,
    matrix: &MatrixHandle,
    protocol: &GafimePrecisionLaunchProtocol,
    device_budget_bytes: Option<u64>,
) -> OrchestratorResult<()> {
    let Some(device_budget_bytes) = device_budget_bytes else {
        return Ok(());
    };
    // SAFETY: callers construct `protocol` from the live compiled-plan storage;
    // this query is synchronous and cannot outlive those descriptor borrows.
    if unsafe { backend.execution_device_memory_peak_bytes_v2(matrix, protocol) }?
        .is_some_and(|peak| peak > device_budget_bytes)
    {
        return Err(OrchestratorError::Unsupported(
            "continuous execution device-memory peak exceeds budget.vram_budget_mb",
        ));
    }
    Ok(())
}

fn validate_execution_device_budget<B: ComputeBackend>(
    backend: &mut B,
    matrix: &MatrixHandle,
    protocol: &GafimeLaunchProtocol,
    device_budget_bytes: Option<u64>,
) -> OrchestratorResult<()> {
    let Some(device_budget_bytes) = device_budget_bytes else {
        return Ok(());
    };
    // SAFETY: callers construct `protocol` from the live compiled-plan storage;
    // this query is synchronous and cannot outlive those descriptor borrows.
    if unsafe { backend.execution_device_memory_peak_bytes(matrix, protocol) }?
        .is_some_and(|peak| peak > device_budget_bytes)
    {
        return Err(OrchestratorError::Unsupported(
            "continuous execution device-memory peak exceeds budget.vram_budget_mb",
        ));
    }
    Ok(())
}

fn validate_rank_override(plan: &CompiledPlan, rank: GafimeRankSpec) -> OrchestratorResult<()> {
    if rank.top_k == 0 {
        return Err(OrchestratorError::InvalidPlan(
            "ranked execution requires a non-zero top_k",
        ));
    }
    if !plan.metric_ids().contains(&rank.primary_metric) {
        return Err(OrchestratorError::InvalidPlan(
            "rank primary metric is not in the plan metric set",
        ));
    }
    if rank.include_ties != 0 {
        return Err(OrchestratorError::Unsupported(
            "rank.include_ties is unsupported",
        ));
    }
    Ok(())
}

fn validate_generated_ranked_execution(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
) -> OrchestratorResult<()> {
    validate_rank_override(plan, rank)?;
    if plan.flags() & GAFIME_LAUNCH_FLAG_GRAPH != 0 {
        return Err(OrchestratorError::Unsupported(
            "generated ranked continuous execution does not support graph capture",
        ));
    }
    Ok(())
}

fn effective_ranked_rows(plan: &CompiledPlan, rank: GafimeRankSpec) -> u64 {
    plan.planned_row_count().min(u64::from(rank.top_k))
}

#[derive(Clone, Copy)]
struct StreamedResultMetadata {
    stable_flags: u32,
    graph_replayed: bool,
    backend_private: *mut core::ffi::c_void,
    reserved: [u64; 8],
}

impl StreamedResultMetadata {
    fn new(result: &mut GafimeResultTable) -> Self {
        let stable_flags = result.flags & !GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
        result.flags = stable_flags;
        Self {
            stable_flags,
            graph_replayed: false,
            backend_private: result.backend_private,
            reserved: result.reserved,
        }
    }

    fn bind_batch(&self, result: &mut GafimeResultTable) {
        result.flags = self.stable_flags;
        result.backend_private = self.backend_private;
        result.reserved = self.reserved;
    }

    fn observe_batch(&mut self, result: &GafimeResultTable) -> OrchestratorResult<()> {
        if result.flags & !GAFIME_RESULT_FLAG_GRAPH_REPLAYED != self.stable_flags {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed non-mergeable result flags",
            ));
        }
        if result.backend_private != self.backend_private {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed result backend_private",
            ));
        }
        if result.reserved != self.reserved {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed reserved result metadata",
            ));
        }
        self.graph_replayed |= (result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0;
        Ok(())
    }

    fn finish(self, result: &mut GafimeResultTable) {
        result.flags = self.stable_flags;
        if self.graph_replayed {
            result.flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
        }
        result.backend_private = self.backend_private;
        result.reserved = self.reserved;
    }
}

#[derive(Clone, Copy)]
struct StreamedResultMetadataF64 {
    stable_flags: u32,
    graph_replayed: bool,
    backend_private: *mut core::ffi::c_void,
    reserved: [u64; 8],
}

impl StreamedResultMetadataF64 {
    fn new(result: &mut GafimeResultTableF64) -> Self {
        let stable_flags = result.flags & !GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
        result.flags = stable_flags;
        Self {
            stable_flags,
            graph_replayed: false,
            backend_private: result.backend_private,
            reserved: result.reserved,
        }
    }

    fn bind_batch(&self, result: &mut GafimeResultTableF64) {
        result.flags = self.stable_flags;
        result.backend_private = self.backend_private;
        result.reserved = self.reserved;
    }

    fn observe_batch(&mut self, result: &GafimeResultTableF64) -> OrchestratorResult<()> {
        if result.flags & !GAFIME_RESULT_FLAG_GRAPH_REPLAYED != self.stable_flags {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed non-mergeable f64 result flags",
            ));
        }
        if result.backend_private != self.backend_private {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed f64 result backend_private",
            ));
        }
        if result.reserved != self.reserved {
            return Err(OrchestratorError::Unsupported(
                "streamed backend changed reserved f64 result metadata",
            ));
        }
        self.graph_replayed |= (result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0;
        Ok(())
    }

    fn finish(self, result: &mut GafimeResultTableF64) {
        result.flags = self.stable_flags;
        if self.graph_replayed {
            result.flags |= GAFIME_RESULT_FLAG_GRAPH_REPLAYED;
        }
        result.backend_private = self.backend_private;
        result.reserved = self.reserved;
    }
}

#[derive(Debug)]
struct OwnedBatchResultTable {
    raw: GafimeResultTable,
    combo_indices: Vec<u32>,
    metric_values: Vec<f32>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl OwnedBatchResultTable {
    fn new(capacity: u64, max_arity: u32, metric_count: u32) -> OrchestratorResult<Self> {
        let capacity = usize::try_from(capacity).map_err(|_| {
            OrchestratorError::InvalidPlan("streamed top-k capacity exceeds the host address space")
        })?;
        let combo_words =
            capacity
                .checked_mul(max_arity as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed top-k combo capacity overflows",
                ))?;
        let metric_values =
            capacity
                .checked_mul(metric_count as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed top-k metric capacity overflows",
                ))?;
        let mut table = Self {
            raw: GafimeResultTable {
                capacity: capacity as u64,
                max_arity,
                metric_count,
                ..Default::default()
            },
            combo_indices: vec![u32::MAX; combo_words],
            metric_values: vec![f32::NAN; metric_values],
            ranks: vec![u32::MAX; capacity],
            families: vec![u32::MAX; capacity],
            candidate_ids: vec![u64::MAX; capacity],
            row_flags: vec![u32::MAX; capacity],
        };
        table.rebind();
        Ok(table)
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

#[derive(Debug)]
struct OwnedBatchResultTableF64 {
    raw: GafimeResultTableF64,
    combo_indices: Vec<u32>,
    metric_values: Vec<f64>,
    ranks: Vec<u32>,
    families: Vec<u32>,
    candidate_ids: Vec<u64>,
    row_flags: Vec<u32>,
}

impl OwnedBatchResultTableF64 {
    fn new(capacity: u64, max_arity: u32, metric_count: u32) -> OrchestratorResult<Self> {
        let capacity = usize::try_from(capacity).map_err(|_| {
            OrchestratorError::InvalidPlan("streamed top-k capacity exceeds the host address space")
        })?;
        let combo_words =
            capacity
                .checked_mul(max_arity as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed top-k combo capacity overflows",
                ))?;
        let metric_values =
            capacity
                .checked_mul(metric_count as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "streamed top-k metric capacity overflows",
                ))?;
        let mut table = Self {
            raw: GafimeResultTableF64 {
                capacity: capacity as u64,
                max_arity,
                metric_count,
                ..Default::default()
            },
            combo_indices: vec![u32::MAX; combo_words],
            metric_values: vec![f64::NAN; metric_values],
            ranks: vec![u32::MAX; capacity],
            families: vec![u32::MAX; capacity],
            candidate_ids: vec![u64::MAX; capacity],
            row_flags: vec![u32::MAX; capacity],
        };
        table.rebind();
        Ok(table)
    }

    fn rebind(&mut self) {
        self.raw.combo_indices = self.combo_indices.as_mut_ptr();
        self.raw.metric_values = self.metric_values.as_mut_ptr();
        self.raw.ranks = self.ranks.as_mut_ptr();
        self.raw.families = self.families.as_mut_ptr();
        self.raw.candidate_ids = self.candidate_ids.as_mut_ptr();
        self.raw.row_flags = self.row_flags.as_mut_ptr();
    }
}

#[derive(Debug)]
struct StreamedRankedRow {
    score: f32,
    candidate_id: u64,
    combo: Vec<u32>,
    metrics: Vec<f32>,
    family: u32,
    row_flags: u32,
}

#[derive(Debug)]
struct StreamedRankedRowF64 {
    score: f64,
    candidate_id: u64,
    combo: Vec<u32>,
    metrics: Vec<f64>,
    family: u32,
    row_flags: u32,
}

fn consider_ranked_row(
    rows: &mut Vec<StreamedRankedRow>,
    row: StreamedRankedRow,
    top_k: usize,
    descending: bool,
) {
    if top_k == 0 {
        return;
    }
    let insertion = rows.partition_point(|current| {
        ranked_row_order(current, &row, descending) == core::cmp::Ordering::Less
    });
    if rows.len() == top_k {
        if insertion == top_k {
            return;
        }
        rows.pop();
    }
    rows.insert(insertion, row);
}

fn ranked_row_order(
    left: &StreamedRankedRow,
    right: &StreamedRankedRow,
    descending: bool,
) -> core::cmp::Ordering {
    let score_order = left
        .score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal);
    let score_order = if descending {
        score_order.reverse()
    } else {
        score_order
    };
    score_order.then_with(|| left.candidate_id.cmp(&right.candidate_id))
}

fn consider_ranked_row_f64(
    rows: &mut Vec<StreamedRankedRowF64>,
    row: StreamedRankedRowF64,
    top_k: usize,
    descending: bool,
) {
    if top_k == 0 {
        return;
    }
    let insertion = rows.partition_point(|current| {
        ranked_row_order_f64(current, &row, descending) == core::cmp::Ordering::Less
    });
    if rows.len() == top_k {
        if insertion == top_k {
            return;
        }
        rows.pop();
    }
    rows.insert(insertion, row);
}

fn ranked_row_order_f64(
    left: &StreamedRankedRowF64,
    right: &StreamedRankedRowF64,
    descending: bool,
) -> core::cmp::Ordering {
    let score_order = left
        .score
        .partial_cmp(&right.score)
        .unwrap_or(core::cmp::Ordering::Equal);
    let score_order = if descending {
        score_order.reverse()
    } else {
        score_order
    };
    score_order.then_with(|| left.candidate_id.cmp(&right.candidate_id))
}

fn validate_ranked_result_table(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
    result: &GafimeResultTable,
) -> OrchestratorResult<()> {
    if result.abi_version != GAFIME_ABI_VERSION {
        return Err(OrchestratorError::InvalidPlan(
            "ranked result table ABI version mismatch",
        ));
    }
    let required_rows = effective_ranked_rows(plan, rank);
    if result.capacity < required_rows {
        return Err(OrchestratorError::InvalidPlan(
            "result table capacity is smaller than ranked plan rows",
        ));
    }
    if result.max_arity < plan.max_arity() || result.metric_count < plan.metric_count() {
        return Err(OrchestratorError::InvalidPlan(
            "result table shape is smaller than the ranked plan",
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
            "ranked result table has null output buffers",
        ));
    }
    Ok(())
}

fn validate_ranked_result_table_f64(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
    result: &GafimeResultTableF64,
) -> OrchestratorResult<()> {
    if result.abi_version != GAFIME_PRECISION_ABI_VERSION {
        return Err(OrchestratorError::InvalidPlan(
            "ranked f64 result table ABI version mismatch",
        ));
    }
    let required_rows = effective_ranked_rows(plan, rank);
    if result.capacity < required_rows {
        return Err(OrchestratorError::InvalidPlan(
            "f64 result table capacity is smaller than ranked plan rows",
        ));
    }
    if result.max_arity < plan.max_arity() || result.metric_count < plan.metric_count() {
        return Err(OrchestratorError::InvalidPlan(
            "f64 result table shape is smaller than the ranked plan",
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
            "ranked f64 result table has null output buffers",
        ));
    }
    Ok(())
}

fn write_ranked_rows(
    result: &mut GafimeResultTable,
    rows: &[StreamedRankedRow],
) -> OrchestratorResult<()> {
    for (rank, row) in rows.iter().enumerate() {
        let combo_base =
            rank.checked_mul(result.max_arity as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "ranked result combo offset overflows",
                ))?;
        let metric_base = rank.checked_mul(result.metric_count as usize).ok_or(
            OrchestratorError::InvalidPlan("ranked result metric offset overflows"),
        )?;
        // SAFETY: validate_ranked_result_table checked every output pointer,
        // capacity, stride, and total row requirement before selection began.
        // The selected rows are bounded by that validated requirement, and the
        // result-table owner guarantees each ABI buffer covers its declared
        // capacity and stride.
        unsafe {
            for slot in 0..result.max_arity as usize {
                *result.combo_indices.add(combo_base + slot) =
                    row.combo.get(slot).copied().unwrap_or(u32::MAX);
            }
            for metric in 0..result.metric_count as usize {
                *result.metric_values.add(metric_base + metric) =
                    row.metrics.get(metric).copied().unwrap_or(0.0);
            }
            *result.ranks.add(rank) = rank as u32;
            *result.families.add(rank) = row.family;
            *result.candidate_ids.add(rank) = row.candidate_id;
            *result.row_flags.add(rank) = row.row_flags;
        }
    }
    result.row_count = rows.len() as u64;
    Ok(())
}

fn write_ranked_rows_f64(
    result: &mut GafimeResultTableF64,
    rows: &[StreamedRankedRowF64],
) -> OrchestratorResult<()> {
    for (rank, row) in rows.iter().enumerate() {
        let combo_base =
            rank.checked_mul(result.max_arity as usize)
                .ok_or(OrchestratorError::InvalidPlan(
                    "ranked f64 result combo offset overflows",
                ))?;
        let metric_base = rank.checked_mul(result.metric_count as usize).ok_or(
            OrchestratorError::InvalidPlan("ranked f64 result metric offset overflows"),
        )?;
        // SAFETY: validate_ranked_result_table_f64 checked every output pointer,
        // capacity, stride, and total row requirement before selection began.
        unsafe {
            for slot in 0..result.max_arity as usize {
                *result.combo_indices.add(combo_base + slot) =
                    row.combo.get(slot).copied().unwrap_or(u32::MAX);
            }
            for metric in 0..result.metric_count as usize {
                *result.metric_values.add(metric_base + metric) =
                    row.metrics.get(metric).copied().unwrap_or(0.0);
            }
            *result.ranks.add(rank) = rank as u32;
            *result.families.add(rank) = row.family;
            *result.candidate_ids.add(rank) = row.candidate_id;
            *result.row_flags.add(rank) = row.row_flags;
        }
    }
    result.row_count = rows.len() as u64;
    Ok(())
}

pub fn continuous_backend_kind(config: &EngineConfig) -> OrchestratorResult<BackendKind> {
    match config.backend_kind {
        0 | GAFIME_BACKEND_CPU => Ok(GAFIME_BACKEND_CPU),
        GAFIME_BACKEND_CUDA => Ok(GAFIME_BACKEND_CUDA),
        GAFIME_BACKEND_ROCM => Ok(GAFIME_BACKEND_ROCM),
        GAFIME_BACKEND_METAL => Ok(GAFIME_BACKEND_METAL),
        _ => Err(OrchestratorError::Unsupported(
            "continuous v1 execution currently supports CPU, CUDA, ROCm, and Metal",
        )),
    }
}

pub fn prepare_continuous_execution(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
) -> OrchestratorResult<PreparedContinuousExecution> {
    let backend_kind = continuous_backend_kind(config)?;
    let plan = build_continuous_plan(continuous_plan_request(
        config,
        rows,
        cols,
        backend_kind,
        GafimeRankSpec::default(),
    ))?;
    prepare_continuous_plan(config, rows, cols, backend_kind, plan, true)
}

pub fn prepare_ranked_continuous_execution(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    rank: GafimeRankSpec,
) -> OrchestratorResult<PreparedContinuousExecution> {
    let backend_kind = continuous_backend_kind(config)?;
    let plan = build_continuous_plan(continuous_plan_request(
        config,
        rows,
        cols,
        backend_kind,
        rank,
    ))?;
    prepare_continuous_plan(config, rows, cols, backend_kind, plan, true)
}

pub fn prepare_continuous_execution_for_feature_orders(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    unary_features: &[u32],
    higher_features: &[u32],
    include_unary: bool,
    include_permutations: bool,
) -> OrchestratorResult<PreparedContinuousExecution> {
    let backend_kind = continuous_backend_kind(config)?;
    let plan = build_continuous_plan_for_feature_orders(
        continuous_plan_request(config, rows, cols, backend_kind, GafimeRankSpec::default()),
        unary_features,
        higher_features,
        include_unary,
    )?;
    prepare_continuous_plan(config, rows, cols, backend_kind, plan, include_permutations)
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_ranked_continuous_execution_for_feature_orders(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    unary_features: &[u32],
    higher_features: &[u32],
    include_unary: bool,
    include_permutations: bool,
    rank: GafimeRankSpec,
) -> OrchestratorResult<PreparedContinuousExecution> {
    let backend_kind = continuous_backend_kind(config)?;
    let plan = build_continuous_plan_for_feature_orders(
        continuous_plan_request(config, rows, cols, backend_kind, rank),
        unary_features,
        higher_features,
        include_unary,
    )?;
    prepare_continuous_plan(config, rows, cols, backend_kind, plan, include_permutations)
}

fn continuous_plan_request(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    backend_kind: BackendKind,
    rank: GafimeRankSpec,
) -> ContinuousPlanRequest {
    ContinuousPlanRequest {
        precision: config.precision,
        backend_kind,
        n_samples: rows,
        n_features: cols,
        max_arity: config.budget.max_comb_size,
        max_combinations_per_arity: config.budget.max_combinations_per_k,
        metric_ids: config.metric_ids.clone(),
        mi_bins: config.mi_bins,
        rank,
    }
}

fn prepare_continuous_plan(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
    backend_kind: BackendKind,
    mut plan: CompiledPlan,
    include_permutations: bool,
) -> OrchestratorResult<PreparedContinuousExecution> {
    if include_permutations && backend_kind == GAFIME_BACKEND_CUDA && config.permutation_tests > 0 {
        plan = plan.with_permutations(GafimePermutationSchedule {
            permutation_count: config.permutation_tests,
            seed: config.random_seed,
            ..Default::default()
        });
    }
    // Opt-in fixed-bin MI approximation backend (CPU only; the GPU always uses
    // fixed bins). Carried as a launch flag the CPU backend reads.
    let mut flags = plan.flags();
    if config.mi_approximate {
        flags |= GAFIME_LAUNCH_FLAG_MI_APPROX;
    }
    if config.graph_requested {
        flags |= GAFIME_LAUNCH_FLAG_GRAPH;
    }
    if flags != plan.flags() {
        plan = plan.with_flags(flags);
    }
    plan.validate()?;
    if plan.uses_generated_descriptors() {
        if plan.rank().top_k == 0 {
            return Err(OrchestratorError::Unsupported(
                "generated public execution requires an explicit report rank cap",
            ));
        }
        validate_generated_ranked_execution(&plan, plan.rank())?;
    }

    // VRAM budget enforcement: fail fast with a clear error instead
    // of OOMing the device when the resident plan would exceed the configured
    // budget. Applies to the GPU backends only (the CPU engine holds no VRAM).
    let device_budget_bytes = if matches!(
        backend_kind,
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
    ) && config.budget.vram_budget_mb > 0
    {
        let footprint = continuous_plan_device_footprint_bytes(rows, cols, &plan, config.precision);
        let budget_bytes = config.budget.vram_budget_mb.saturating_mul(1024 * 1024);
        if footprint > budget_bytes {
            return Err(OrchestratorError::Unsupported(
                "continuous plan device footprint exceeds budget.vram_budget_mb",
            ));
        }
        Some(budget_bytes)
    } else {
        None
    };

    let schedule = ContinuousSchedule::for_plan(&plan)?;
    Ok(PreparedContinuousExecution {
        plan,
        schedule,
        precision: config.precision,
        descriptor_generation: next_descriptor_generation(),
        rows,
        cols,
        device_budget_bytes,
    })
}

/// Estimate peak resident device buffers for this plan. Generated ranked plans
/// retain the maximum capacity reached across descriptor batches; metric values
/// therefore scale with batch rows, while result/gather buffers scale with K.
pub fn continuous_plan_device_footprint_bytes(
    rows: u64,
    cols: u32,
    plan: &CompiledPlan,
    precision: PrecisionProfile,
) -> u64 {
    continuous_plan_device_footprint_bytes_for_rank(rows, cols, plan, precision, plan.rank())
}

fn continuous_plan_device_footprint_bytes_for_rank(
    rows: u64,
    cols: u32,
    plan: &CompiledPlan,
    precision: PrecisionProfile,
    rank: GafimeRankSpec,
) -> u64 {
    let fixed_bytes =
        continuous_matrix_device_footprint_bytes(plan.backend_kind(), precision, rows, cols);
    continuous_launch_sequence_peak_bytes(
        plan.backend_kind(),
        precision,
        fixed_bytes,
        continuous_plan_launch_footprint_shapes(plan, rank, 1),
    )
}

/// Estimate the exact native allocation high-water mark across sequential
/// executions that share one resident matrix. CUDA and ROCm retain geometrically
/// grown buffers and briefly own old plus replacement allocations. Metal retains
/// only its immutable descriptor cache; result and ranking buffers are scoped to
/// one execution.
pub fn continuous_staged_device_footprint_bytes(stages: &[&PreparedContinuousExecution]) -> u64 {
    let Some(first) = stages.first() else {
        return 0;
    };
    if stages.iter().any(|stage| {
        stage.rows != first.rows
            || stage.cols != first.cols
            || stage.precision != first.precision
            || stage.plan.backend_kind() != first.plan.backend_kind()
    }) {
        return u64::MAX;
    }

    let backend_kind = first.plan.backend_kind();
    let mut shapes = Vec::new();
    for stage in stages {
        shapes.extend(continuous_plan_launch_footprint_shapes(
            &stage.plan,
            stage.plan.rank(),
            stage.descriptor_generation,
        ));
    }

    continuous_launch_sequence_peak_bytes(
        backend_kind,
        first.precision,
        continuous_matrix_device_footprint_bytes(
            backend_kind,
            first.precision,
            first.rows,
            first.cols,
        ),
        shapes,
    )
}

/// Estimate the resident device-memory footprint (bytes) of a continuous plan:
/// the feature matrix, target, column means and statistics, combo-index and
/// metric-id buffers, and unranked metric output. Saturates on huge plans.
pub fn continuous_device_footprint_bytes(
    rows: u64,
    cols: u32,
    metric_count: u64,
    planned_rows: u64,
    combo_slots: u64,
) -> u64 {
    continuous_matrix_device_footprint_bytes(
        GAFIME_BACKEND_CUDA,
        PrecisionProfile::Fp32,
        rows,
        cols,
    )
    .saturating_add(metric_count.saturating_mul(4))
    .saturating_add(continuous_launch_device_footprint_bytes(
        GAFIME_BACKEND_CUDA,
        PrecisionProfile::Fp32,
        planned_rows,
        combo_slots,
        metric_count,
        0,
        0,
    ))
}

fn continuous_matrix_device_footprint_bytes(
    backend_kind: BackendKind,
    precision: PrecisionProfile,
    rows: u64,
    cols: u32,
) -> u64 {
    const U32_BYTES: u64 = 4;
    const U64_BYTES: u64 = 8;
    let storage_bytes = precision_storage_bytes(precision);
    let accumulation_bytes = precision_accumulation_bytes(precision);
    let mean_bytes = if backend_kind == GAFIME_BACKEND_ROCM {
        accumulation_bytes
    } else {
        storage_bytes
    };
    let base = rows
        .saturating_mul(u64::from(cols))
        .saturating_mul(storage_bytes)
        .saturating_add(rows.saturating_mul(storage_bytes))
        .saturating_add(u64::from(cols).saturating_mul(mean_bytes));
    if backend_kind == GAFIME_BACKEND_METAL {
        // Metal allocates the bounded fp32 target-rank cache with the resident
        // matrix, even when the selected metric set does not use Spearman.
        let target_rank_bytes = if rows <= 4_096 {
            rows.saturating_mul(U32_BYTES)
        } else {
            U32_BYTES
        };
        return base.saturating_add(target_rank_bytes);
    }

    let (target_stats_bytes, feature_stats_bytes) = if backend_kind == GAFIME_BACKEND_CUDA {
        if precision == PrecisionProfile::Fp32 {
            (24, 24)
        } else {
            (32, 32)
        }
    } else if precision == PrecisionProfile::Fp32 {
        (16, 16)
    } else {
        (24, 24)
    };
    // CUDA attempts this bounded cache for every ABI 1.1 resident matrix,
    // independent of the current metric selection. The cache is not allocated
    // above the native ceiling. Count the attempted allocation conservatively
    // before payload discovery; the exact native peak later reflects whether
    // the optional allocation succeeded.
    let spearman_rank_bytes = if backend_kind == GAFIME_BACKEND_CUDA && rows <= 4_096 {
        rows.saturating_mul(U64_BYTES)
    } else {
        0
    };
    base.saturating_add(target_stats_bytes)
        .saturating_add(u64::from(cols).saturating_mul(feature_stats_bytes))
        .saturating_add(spearman_rank_bytes)
}

fn continuous_plan_launch_footprint_shapes(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
    descriptor_generation: u64,
) -> Vec<LaunchFootprintShape> {
    if plan.uses_generated_descriptors() && rank.top_k > 0 {
        generated_ranked_launch_footprint_shapes(plan, rank)
    } else {
        let mut shape = continuous_launch_footprint_shape(
            plan.backend_kind(),
            plan.planned_row_count(),
            plan.logical_descriptor_words(),
            u64::from(plan.metric_count()),
            u64::from(rank.top_k),
            plan.chunks().len() as u64,
        );
        shape.descriptor_generation = descriptor_generation;
        vec![shape]
    }
}

fn generated_ranked_launch_footprint_shapes(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
) -> Vec<LaunchFootprintShape> {
    let effective_top_k = effective_ranked_rows(plan, rank);
    let max_words = DEFAULT_DESCRIPTOR_BATCH_WORDS as u64;
    plan.chunks()
        .iter()
        .filter_map(|chunk| {
            let arity = u64::from(chunk.arity);
            let batch_rows = chunk.combo_count.min((max_words / arity).max(1));
            if batch_rows == 0 {
                return None;
            }
            let descriptor_words = batch_rows.saturating_mul(arity);
            let local_top_k = batch_rows.min(effective_top_k);
            let partial_items =
                topk_partial_block_count(plan.backend_kind(), batch_rows, local_top_k)
                    .saturating_mul(local_top_k);
            Some(LaunchFootprintShape {
                launch_rows: batch_rows,
                descriptor_words,
                metric_count: u64::from(plan.metric_count()),
                effective_top_k: local_top_k,
                partial_items,
                chunk_count: 1,
                // Streamed protocols explicitly clear immutable/generation state.
                descriptor_generation: 0,
            })
        })
        .collect()
}

fn continuous_launch_device_footprint_bytes(
    backend_kind: BackendKind,
    precision: PrecisionProfile,
    launch_rows: u64,
    descriptor_words: u64,
    metric_count: u64,
    requested_top_k: u64,
    chunk_count: u64,
) -> u64 {
    let shape = continuous_launch_footprint_shape(
        backend_kind,
        launch_rows,
        descriptor_words,
        metric_count,
        requested_top_k,
        chunk_count,
    );
    ranked_launch_device_footprint_bytes(backend_kind, precision, shape)
}

fn continuous_launch_footprint_shape(
    backend_kind: BackendKind,
    launch_rows: u64,
    descriptor_words: u64,
    metric_count: u64,
    requested_top_k: u64,
    chunk_count: u64,
) -> LaunchFootprintShape {
    let effective_top_k = launch_rows.min(requested_top_k);
    let partial_items = topk_partial_block_count(backend_kind, launch_rows, effective_top_k)
        .saturating_mul(effective_top_k);
    LaunchFootprintShape {
        launch_rows,
        descriptor_words,
        metric_count,
        effective_top_k,
        partial_items,
        chunk_count,
        descriptor_generation: 0,
    }
}

#[derive(Clone, Copy, Default)]
struct LaunchFootprintShape {
    launch_rows: u64,
    descriptor_words: u64,
    metric_count: u64,
    effective_top_k: u64,
    partial_items: u64,
    chunk_count: u64,
    /// A non-zero generation denotes an immutable, cacheable descriptor set.
    descriptor_generation: u64,
}

#[derive(Clone, Copy, Default)]
struct LaunchBufferCapacities {
    descriptor_bytes: u64,
    metric_value_bytes: u64,
    selected_index_bytes: u64,
    selected_metric_bytes: u64,
    partial_score_bytes: u64,
    partial_index_bytes: u64,
    metal_chunk_bytes: u64,
    metal_launch_info_bytes: u64,
    metal_rank_info_bytes: u64,
}

impl LaunchBufferCapacities {
    fn total_bytes(self) -> u64 {
        self.descriptor_bytes
            .saturating_add(self.metric_value_bytes)
            .saturating_add(self.selected_index_bytes)
            .saturating_add(self.selected_metric_bytes)
            .saturating_add(self.partial_score_bytes)
            .saturating_add(self.partial_index_bytes)
            .saturating_add(self.metal_chunk_bytes)
            .saturating_add(self.metal_launch_info_bytes)
            .saturating_add(self.metal_rank_info_bytes)
    }
}

fn ranked_launch_device_footprint_bytes(
    backend_kind: BackendKind,
    precision: PrecisionProfile,
    shape: LaunchFootprintShape,
) -> u64 {
    ranked_launch_buffer_capacities(backend_kind, precision, shape).total_bytes()
}

fn ranked_launch_buffer_capacities(
    backend_kind: BackendKind,
    precision: PrecisionProfile,
    shape: LaunchFootprintShape,
) -> LaunchBufferCapacities {
    const U32_BYTES: u64 = 4;
    let result_bytes = precision_result_bytes(precision);

    let descriptor_bytes = shape.descriptor_words.saturating_mul(U32_BYTES);
    let metric_value_bytes = shape
        .launch_rows
        .saturating_mul(shape.metric_count)
        .saturating_mul(result_bytes);
    let selected_index_bytes = shape.effective_top_k.saturating_mul(U32_BYTES);
    let selected_metric_bytes = shape
        .effective_top_k
        .saturating_mul(shape.metric_count)
        .saturating_mul(result_bytes);
    let partial_score_bytes = shape.partial_items.saturating_mul(result_bytes);
    let partial_index_bytes = shape.partial_items.saturating_mul(U32_BYTES);
    let metal = backend_kind == GAFIME_BACKEND_METAL;
    LaunchBufferCapacities {
        descriptor_bytes,
        metric_value_bytes,
        selected_index_bytes,
        selected_metric_bytes,
        partial_score_bytes,
        partial_index_bytes,
        metal_chunk_bytes: if metal {
            shape.chunk_count.saturating_mul(40)
        } else {
            0
        },
        metal_launch_info_bytes: if metal { 24 } else { 0 },
        metal_rank_info_bytes: if metal && shape.effective_top_k != 0 {
            24
        } else {
            0
        },
    }
}

#[derive(Default)]
struct RetainedLaunchCapacities {
    descriptor_words: u64,
    metric_ids: u64,
    metric_values: u64,
    selected_indices: u64,
    selected_metric_values: u64,
    partial_scores: u64,
    partial_indices: u64,
}

struct DeviceMemoryPeakSimulation {
    resident_bytes: u64,
    peak_bytes: u64,
}

impl DeviceMemoryPeakSimulation {
    fn new(resident_bytes: u64) -> Self {
        Self {
            resident_bytes,
            peak_bytes: resident_bytes,
        }
    }

    fn observe_transient(&mut self, transient_bytes: u64) {
        self.peak_bytes = self
            .peak_bytes
            .max(self.resident_bytes.saturating_add(transient_bytes));
    }

    fn replace_resident(&mut self, old_bytes: u64, next_bytes: u64) {
        self.resident_bytes = self
            .resident_bytes
            .checked_sub(old_bytes)
            .map_or(u64::MAX, |remaining| remaining.saturating_add(next_bytes));
        self.peak_bytes = self.peak_bytes.max(self.resident_bytes);
    }
}

fn allocation_bytes(capacity: u64, element_bytes: u64) -> u64 {
    capacity.saturating_mul(element_bytes)
}

fn next_allocation_capacity(capacity: u64, required: u64, element_bytes: u64) -> u64 {
    if required <= capacity {
        return capacity;
    }
    let max_capacity = (usize::MAX as u64) / element_bytes;
    if required > max_capacity {
        return u64::MAX;
    }
    let grown_capacity = if capacity > max_capacity / 2 {
        max_capacity
    } else {
        capacity.saturating_mul(2)
    };
    required.max(if capacity == 0 {
        required
    } else {
        grown_capacity
    })
}

fn simulate_buffer_growth(
    simulation: &mut DeviceMemoryPeakSimulation,
    capacity: &mut u64,
    required: u64,
    element_bytes: u64,
) {
    if required <= *capacity {
        return;
    }
    let next_capacity = next_allocation_capacity(*capacity, required, element_bytes);
    let old_bytes = allocation_bytes(*capacity, element_bytes);
    let next_bytes = allocation_bytes(next_capacity, element_bytes);
    simulation.observe_transient(next_bytes);
    simulation.replace_resident(old_bytes, next_bytes);
    *capacity = next_capacity;
}

#[allow(clippy::too_many_arguments)]
fn simulate_buffer_pair_growth(
    simulation: &mut DeviceMemoryPeakSimulation,
    first_capacity: &mut u64,
    first_required: u64,
    first_element_bytes: u64,
    second_capacity: &mut u64,
    second_required: u64,
    second_element_bytes: u64,
) {
    let first_next = next_allocation_capacity(*first_capacity, first_required, first_element_bytes);
    let second_next =
        next_allocation_capacity(*second_capacity, second_required, second_element_bytes);
    let mut transient_bytes = 0u64;
    if first_required > *first_capacity {
        transient_bytes =
            transient_bytes.saturating_add(allocation_bytes(first_next, first_element_bytes));
    }
    if second_required > *second_capacity {
        transient_bytes =
            transient_bytes.saturating_add(allocation_bytes(second_next, second_element_bytes));
    }
    simulation.observe_transient(transient_bytes);
    if first_required > *first_capacity {
        simulation.replace_resident(
            allocation_bytes(*first_capacity, first_element_bytes),
            allocation_bytes(first_next, first_element_bytes),
        );
        *first_capacity = first_next;
    }
    if second_required > *second_capacity {
        simulation.replace_resident(
            allocation_bytes(*second_capacity, second_element_bytes),
            allocation_bytes(second_next, second_element_bytes),
        );
        *second_capacity = second_next;
    }
}

fn retained_launch_sequence_peak_bytes(
    precision: PrecisionProfile,
    fixed_bytes: u64,
    shapes: impl IntoIterator<Item = LaunchFootprintShape>,
) -> u64 {
    const U32_BYTES: u64 = 4;
    let result_bytes = precision_result_bytes(precision);
    let mut simulation = DeviceMemoryPeakSimulation::new(fixed_bytes);
    let mut capacities = RetainedLaunchCapacities::default();

    for shape in shapes {
        simulate_buffer_growth(
            &mut simulation,
            &mut capacities.metric_values,
            shape.launch_rows.saturating_mul(shape.metric_count),
            result_bytes,
        );
        simulate_buffer_pair_growth(
            &mut simulation,
            &mut capacities.descriptor_words,
            shape.descriptor_words,
            U32_BYTES,
            &mut capacities.metric_ids,
            shape.metric_count,
            U32_BYTES,
        );
        if shape.effective_top_k == 0 {
            continue;
        }
        simulate_buffer_growth(
            &mut simulation,
            &mut capacities.selected_indices,
            shape.effective_top_k,
            U32_BYTES,
        );
        simulate_buffer_growth(
            &mut simulation,
            &mut capacities.partial_scores,
            shape.partial_items,
            result_bytes,
        );
        simulate_buffer_growth(
            &mut simulation,
            &mut capacities.partial_indices,
            shape.partial_items,
            U32_BYTES,
        );
        simulate_buffer_growth(
            &mut simulation,
            &mut capacities.selected_metric_values,
            shape.effective_top_k.saturating_mul(shape.metric_count),
            result_bytes,
        );
    }
    simulation.peak_bytes
}

fn metal_launch_sequence_peak_bytes(
    precision: PrecisionProfile,
    fixed_bytes: u64,
    shapes: impl IntoIterator<Item = LaunchFootprintShape>,
) -> u64 {
    const U32_BYTES: u64 = 4;
    const METAL_CHUNK_BYTES: u64 = 40;
    const METAL_LAUNCH_INFO_BYTES: u64 = 24;
    const METAL_RANK_INFO_BYTES: u64 = 24;
    let result_bytes = precision_result_bytes(precision);
    let mut peak_bytes = fixed_bytes;
    let mut cached_descriptor_bytes = 0u64;
    let mut cached_descriptor_generation = 0u64;

    for shape in shapes {
        let descriptor_bytes = shape
            .descriptor_words
            .saturating_mul(U32_BYTES)
            .saturating_add(shape.metric_count.saturating_mul(U32_BYTES))
            .saturating_add(shape.chunk_count.saturating_mul(METAL_CHUNK_BYTES))
            .saturating_add(METAL_LAUNCH_INFO_BYTES);
        let cacheable = shape.descriptor_generation != 0;
        let descriptors_resident = cacheable
            && cached_descriptor_generation == shape.descriptor_generation
            && cached_descriptor_bytes != 0;
        let resident_bytes = fixed_bytes.saturating_add(cached_descriptor_bytes);
        let execution_resident_bytes = if descriptors_resident {
            resident_bytes
        } else {
            peak_bytes = peak_bytes.max(resident_bytes.saturating_add(descriptor_bytes));
            if cacheable {
                fixed_bytes.saturating_add(descriptor_bytes)
            } else {
                resident_bytes.saturating_add(descriptor_bytes)
            }
        };

        let mut runtime_bytes = execution_resident_bytes.saturating_add(
            shape
                .launch_rows
                .saturating_mul(shape.metric_count)
                .saturating_mul(result_bytes),
        );
        if shape.effective_top_k != 0 {
            runtime_bytes = runtime_bytes
                .saturating_add(METAL_RANK_INFO_BYTES)
                .saturating_add(shape.effective_top_k.saturating_mul(U32_BYTES))
                .saturating_add(
                    shape
                        .effective_top_k
                        .saturating_mul(shape.metric_count)
                        .saturating_mul(result_bytes),
                )
                .saturating_add(shape.partial_items.saturating_mul(result_bytes))
                .saturating_add(shape.partial_items.saturating_mul(U32_BYTES));
        }
        peak_bytes = peak_bytes.max(runtime_bytes);
        if cacheable && !descriptors_resident {
            cached_descriptor_bytes = descriptor_bytes;
            cached_descriptor_generation = shape.descriptor_generation;
        }
    }
    peak_bytes
}

fn continuous_launch_sequence_peak_bytes(
    backend_kind: BackendKind,
    precision: PrecisionProfile,
    fixed_bytes: u64,
    shapes: impl IntoIterator<Item = LaunchFootprintShape>,
) -> u64 {
    if backend_kind == GAFIME_BACKEND_METAL {
        metal_launch_sequence_peak_bytes(precision, fixed_bytes, shapes)
    } else {
        retained_launch_sequence_peak_bytes(precision, fixed_bytes, shapes)
    }
}

const fn precision_storage_bytes(precision: PrecisionProfile) -> u64 {
    match precision {
        PrecisionProfile::Fp32 | PrecisionProfile::Mixed => 4,
        PrecisionProfile::Fp64 => 8,
    }
}

const fn precision_accumulation_bytes(precision: PrecisionProfile) -> u64 {
    match precision {
        PrecisionProfile::Fp32 => 4,
        PrecisionProfile::Mixed | PrecisionProfile::Fp64 => 8,
    }
}

const fn precision_result_bytes(precision: PrecisionProfile) -> u64 {
    precision_accumulation_bytes(precision)
}

fn topk_partial_block_count(backend_kind: BackendKind, row_count: u64, top_k: u64) -> u64 {
    // Before payload discovery the CUDA device class is unknown. Use the
    // smallest supported launch geometry so the forecast is conservative for
    // pre-Ampere devices; ROCm's distributed kernels use 256 threads.
    const CUDA_TOPK_THREADS_PER_BLOCK: u64 = 128;
    const ROCM_TOPK_THREADS_PER_BLOCK: u64 = 256;
    const METAL_TOPK_THREADS_PER_BLOCK: u64 = 64;
    const TOPK_MAX_PARTIAL_BLOCKS: u64 = 4096;

    if row_count == 0 || top_k == 0 {
        return 0;
    }
    let threads_per_block = match backend_kind {
        GAFIME_BACKEND_CUDA => CUDA_TOPK_THREADS_PER_BLOCK,
        GAFIME_BACKEND_METAL => METAL_TOPK_THREADS_PER_BLOCK,
        _ => ROCM_TOPK_THREADS_PER_BLOCK,
    };
    let target_blocks = 1 + (row_count - 1) / threads_per_block;
    let storage_blocks = 1 + (row_count - 1) / top_k;
    target_blocks
        .min(storage_blocks)
        .min(TOPK_MAX_PARTIAL_BLOCKS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{
        GafimeArityChunk, GAFIME_BACKEND_METAL, GAFIME_FAMILY_CONTINUOUS, GAFIME_LAUNCH_FLAG_GRAPH,
        GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_R2,
    };

    #[derive(Debug)]
    struct RecordedLaunch {
        flags: u32,
        descriptor_generation: u64,
        combo_indices: Vec<u32>,
        rank_top_k: u32,
        result_capacity: u64,
        result_flags: u32,
        result_backend_private: usize,
        result_reserved: [u64; 8],
        permutation_count: u32,
    }

    #[derive(Default)]
    struct RecordingBackend {
        launch_flags: u32,
        descriptor_generation: u64,
        launches: Vec<RecordedLaunch>,
        output_result_flags: u32,
        replace_backend_private: bool,
        device_memory_peak_bytes: Option<u64>,
        device_memory_preflights: usize,
    }

    impl ComputeBackend for RecordingBackend {
        fn backend_kind(&self) -> BackendKind {
            GAFIME_BACKEND_CPU
        }

        unsafe fn execution_device_memory_peak_bytes(
            &mut self,
            _matrix: &MatrixHandle,
            _protocol: &GafimeLaunchProtocol,
        ) -> OrchestratorResult<Option<u64>> {
            self.device_memory_preflights += 1;
            Ok(self.device_memory_peak_bytes)
        }

        unsafe fn execute(
            &mut self,
            _matrix: &MatrixHandle,
            protocol: &gafime_types::GafimeLaunchProtocol,
            result: &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> {
            self.launch_flags = protocol.flags;
            self.descriptor_generation = protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT];
            // SAFETY: the test backend is called synchronously by the prepared
            // plan, which owns exactly `chunk_count` initialized descriptors.
            let chunks = unsafe {
                core::slice::from_raw_parts(protocol.chunks, protocol.chunk_count as usize)
            };
            // SAFETY: the same prepared plan owns the declared combo-index
            // buffer and keeps it live throughout this recording call.
            let combo_indices = unsafe {
                core::slice::from_raw_parts(
                    protocol.combo_indices.ptr,
                    protocol.combo_indices.len as usize,
                )
            };
            self.launches.push(RecordedLaunch {
                flags: protocol.flags,
                descriptor_generation: self.descriptor_generation,
                combo_indices: combo_indices.to_vec(),
                rank_top_k: protocol.rank.top_k,
                result_capacity: result.capacity,
                result_flags: result.flags,
                result_backend_private: result.backend_private as usize,
                result_reserved: result.reserved,
                permutation_count: protocol.permutations.permutation_count,
            });
            let planned_rows = chunks.iter().map(|chunk| chunk.combo_count).sum::<u64>();
            let descriptor_for_row = |candidate_id: u64| {
                let mut row_offset = 0u64;
                for chunk in chunks {
                    if candidate_id < row_offset + chunk.combo_count {
                        let local_row = candidate_id - row_offset;
                        let descriptor_base = chunk.descriptor_offset as usize
                            + local_row as usize * chunk.arity as usize;
                        return Some((chunk, descriptor_base));
                    }
                    row_offset += chunk.combo_count;
                }
                None
            };
            let score_for_row = |candidate_id: u64| {
                let (chunk, descriptor_base) = descriptor_for_row(candidate_id).unwrap();
                combo_indices[descriptor_base..descriptor_base + chunk.arity as usize]
                    .iter()
                    .map(|&feature| feature as f32)
                    .sum::<f32>()
            };
            let mut selected_rows = (0..planned_rows).collect::<Vec<_>>();
            if protocol.rank.top_k > 0 {
                selected_rows.sort_by(|&left, &right| {
                    let order = score_for_row(left)
                        .partial_cmp(&score_for_row(right))
                        .unwrap();
                    let order = if protocol.rank.descending != 0 {
                        order.reverse()
                    } else {
                        order
                    };
                    order.then_with(|| left.cmp(&right))
                });
                selected_rows.truncate(protocol.rank.top_k as usize);
            }
            let output_rows = selected_rows.len() as u64;
            result.row_count = output_rows;
            result.flags |= self.output_result_flags;
            if self.replace_backend_private {
                result.backend_private = core::ptr::NonNull::<u8>::dangling().as_ptr().cast();
            }
            if output_rows > 0
                && !result.combo_indices.is_null()
                && !result.metric_values.is_null()
                && !result.ranks.is_null()
                && !result.families.is_null()
                && !result.candidate_ids.is_null()
                && !result.row_flags.is_null()
            {
                for (row, &candidate_id) in selected_rows.iter().enumerate() {
                    let (chunk, descriptor_base) = descriptor_for_row(candidate_id).unwrap();
                    let score = score_for_row(candidate_id);
                    // SAFETY: the test result owner allocated each non-null
                    // buffer for its declared capacity and strides. selected_rows
                    // is truncated to top_k, which is bounded by that capacity.
                    unsafe {
                        for slot in 0..result.max_arity as usize {
                            *result
                                .combo_indices
                                .add(row * result.max_arity as usize + slot) =
                                if slot < chunk.arity as usize {
                                    combo_indices[descriptor_base + slot]
                                } else {
                                    u32::MAX
                                };
                        }
                        for metric in 0..result.metric_count as usize {
                            *result
                                .metric_values
                                .add(row * result.metric_count as usize + metric) = score;
                        }
                        *result.ranks.add(row) = row as u32;
                        *result.families.add(row) = GAFIME_FAMILY_CONTINUOUS;
                        *result.candidate_ids.add(row) = candidate_id;
                        *result.row_flags.add(row) = 0;
                    }
                }
            }
            Ok(BackendExecutionStats {
                launched_chunks: chunks.len() as u64,
                graph_replays: u64::from((result.flags & GAFIME_RESULT_FLAG_GRAPH_REPLAYED) != 0),
                rows_written: output_rows,
            })
        }
    }

    struct PanicPrecisionBackend;

    impl PrecisionComputeBackend for PanicPrecisionBackend {
        fn backend_kind(&self) -> BackendKind {
            GAFIME_BACKEND_CPU
        }

        unsafe fn execute_fp32(
            &mut self,
            _matrix: &MatrixHandle,
            _protocol: &GafimePrecisionLaunchProtocol,
            _result: &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> {
            panic!("precision identity validation must run before fp32 dispatch")
        }

        unsafe fn execute_f64(
            &mut self,
            _matrix: &MatrixHandle,
            _protocol: &GafimePrecisionLaunchProtocol,
            _result: &mut GafimeResultTableF64,
        ) -> OrchestratorResult<BackendExecutionStats> {
            panic!("precision identity validation must run before f64 dispatch")
        }
    }

    struct TestResultTable {
        raw: GafimeResultTable,
        combo_indices: Vec<u32>,
        metric_values: Vec<f32>,
        ranks: Vec<u32>,
        families: Vec<u32>,
        candidate_ids: Vec<u64>,
        row_flags: Vec<u32>,
    }

    #[test]
    fn raw_prepared_execution_routes_remain_unsafe_function_items() {
        #[allow(dead_code)]
        #[deny(unused_unsafe)]
        fn require_unsafe_calls(
            prepared: &PreparedContinuousExecution,
            compute: &mut RecordingBackend,
            precision: &mut PanicPrecisionBackend,
            matrix: &MatrixHandle,
            result_f32: &mut GafimeResultTable,
            result_f64: &mut GafimeResultTableF64,
        ) {
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe { prepared.execute(compute, matrix, result_f32) };
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe { prepared.execute_precision_fp32(precision, matrix, result_f32) };
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe { prepared.execute_precision_f64(precision, matrix, result_f64) };
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe {
                prepared.execute_ranked(GafimeRankSpec::default(), compute, matrix, result_f32)
            };
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe {
                prepared.execute_precision_ranked_fp32(
                    GafimeRankSpec::default(),
                    precision,
                    matrix,
                    result_f32,
                )
            };
            // SAFETY: compile-only API assertion; this function is never called.
            let _ = unsafe {
                prepared.execute_precision_ranked_f64(
                    GafimeRankSpec::default(),
                    precision,
                    matrix,
                    result_f64,
                )
            };
        }

        let _: unsafe fn(
            &PreparedContinuousExecution,
            &mut RecordingBackend,
            &MatrixHandle,
            &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute::<RecordingBackend>;
        let _: unsafe fn(
            &PreparedContinuousExecution,
            &mut PanicPrecisionBackend,
            &MatrixHandle,
            &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute_precision_fp32::<PanicPrecisionBackend>;
        let _: unsafe fn(
            &PreparedContinuousExecution,
            &mut PanicPrecisionBackend,
            &MatrixHandle,
            &mut GafimeResultTableF64,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute_precision_f64::<PanicPrecisionBackend>;
        let _: unsafe fn(
            &PreparedContinuousExecution,
            GafimeRankSpec,
            &mut RecordingBackend,
            &MatrixHandle,
            &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute_ranked::<RecordingBackend>;
        let _: unsafe fn(
            &PreparedContinuousExecution,
            GafimeRankSpec,
            &mut PanicPrecisionBackend,
            &MatrixHandle,
            &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute_precision_ranked_fp32::<PanicPrecisionBackend>;
        let _: unsafe fn(
            &PreparedContinuousExecution,
            GafimeRankSpec,
            &mut PanicPrecisionBackend,
            &MatrixHandle,
            &mut GafimeResultTableF64,
        ) -> OrchestratorResult<BackendExecutionStats> =
            PreparedContinuousExecution::execute_precision_ranked_f64::<PanicPrecisionBackend>;
    }

    impl TestResultTable {
        fn new(capacity: u64, max_arity: u32, metric_count: u32) -> Self {
            let capacity = capacity as usize;
            let mut table = Self {
                raw: GafimeResultTable {
                    capacity: capacity as u64,
                    max_arity,
                    metric_count,
                    ..Default::default()
                },
                combo_indices: vec![u32::MAX; capacity * max_arity as usize],
                metric_values: vec![f32::NAN; capacity * metric_count as usize],
                ranks: vec![u32::MAX; capacity],
                families: vec![u32::MAX; capacity],
                candidate_ids: vec![u64::MAX; capacity],
                row_flags: vec![u32::MAX; capacity],
            };
            table.raw.combo_indices = table.combo_indices.as_mut_ptr();
            table.raw.metric_values = table.metric_values.as_mut_ptr();
            table.raw.ranks = table.ranks.as_mut_ptr();
            table.raw.families = table.families.as_mut_ptr();
            table.raw.candidate_ids = table.candidate_ids.as_mut_ptr();
            table.raw.row_flags = table.row_flags.as_mut_ptr();
            table
        }
    }

    fn generated_pair_plan(rank: GafimeRankSpec) -> CompiledPlan {
        let features = [4, 0, 5, 2];
        let source = crate::plan::combos::CombinationDescriptorSource::new(&[], &features);
        let mut shape_hint = crate::plan::shapes::default_shape_hint(GAFIME_BACKEND_CPU, 2);
        shape_hint.vendor_hint = 2;
        CompiledPlan::from_combination_parts(
            GAFIME_BACKEND_CPU,
            8,
            6,
            2,
            source,
            vec![GAFIME_METRIC_PEARSON],
            vec![GafimeArityChunk {
                arity: 2,
                family: GAFIME_FAMILY_CONTINUOUS,
                metric_mask: 0,
                shape_hint_index: 0,
                combo_row_offset: 0,
                combo_count: 6,
                local_chunk_id: 0,
                flags: 0,
                descriptor_offset: 0,
                descriptor_count: 6,
            }],
            vec![shape_hint],
            rank,
            GafimePermutationSchedule::default(),
        )
    }

    fn generated_mixed_arity_plan(rank: GafimeRankSpec) -> CompiledPlan {
        let unary_features = [3, 1, 4];
        let higher_features = [4, 0, 3, 1];
        let source = crate::plan::combos::CombinationDescriptorSource::new(
            &unary_features,
            &higher_features,
        );
        let mut unary_shape = crate::plan::shapes::default_shape_hint(GAFIME_BACKEND_CPU, 1);
        unary_shape.vendor_hint = 2;
        let mut pair_shape = crate::plan::shapes::default_shape_hint(GAFIME_BACKEND_CPU, 2);
        pair_shape.vendor_hint = 2;
        CompiledPlan::from_combination_parts(
            GAFIME_BACKEND_CPU,
            8,
            5,
            2,
            source,
            vec![GAFIME_METRIC_PEARSON],
            vec![
                GafimeArityChunk {
                    arity: 1,
                    family: GAFIME_FAMILY_CONTINUOUS,
                    metric_mask: 0,
                    shape_hint_index: 0,
                    combo_row_offset: 0,
                    combo_count: 3,
                    local_chunk_id: 0,
                    flags: 0,
                    descriptor_offset: 0,
                    descriptor_count: 3,
                },
                GafimeArityChunk {
                    arity: 2,
                    family: GAFIME_FAMILY_CONTINUOUS,
                    metric_mask: 0,
                    shape_hint_index: 1,
                    combo_row_offset: 3,
                    combo_count: 6,
                    local_chunk_id: 1,
                    flags: 0,
                    descriptor_offset: 3,
                    descriptor_count: 6,
                },
            ],
            vec![unary_shape, pair_shape],
            rank,
            GafimePermutationSchedule::default(),
        )
    }

    fn prepare_generated_pair_plan(plan: CompiledPlan) -> PreparedContinuousExecution {
        let config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..EngineConfig::default()
        };
        prepare_continuous_plan(&config, 8, 6, GAFIME_BACKEND_CPU, plan, false).unwrap()
    }

    #[test]
    fn default_config_prepares_cpu_continuous_execution() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 3;
        config.budget.max_combinations_per_k = 1_000;

        let prepared = prepare_continuous_execution(&config, 32, 5).unwrap();

        assert_eq!(prepared.plan().protocol().backend_kind, GAFIME_BACKEND_CPU);
        assert_eq!(prepared.plan().protocol().permutations.permutation_count, 0);
        assert_eq!(prepared.result_max_arity(), 3);
        assert_eq!(prepared.result_metric_count(), 2);
        assert_eq!(prepared.result_capacity(), 25);
        assert_eq!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );
    }

    #[test]
    fn precision_execution_rejects_wrong_result_lane_and_profile_identity_before_dispatch() {
        let mut config = EngineConfig {
            precision: PrecisionProfile::Fp32,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;
        let fp32_prepared = prepare_continuous_execution(&config, 8, 2).unwrap();
        let fp32_matrix =
            MatrixHandle::host_with_precision(GAFIME_BACKEND_CPU, PrecisionProfile::Fp32, 8, 2);
        let mut f64_result = GafimeResultTableF64::default();
        let mut backend = PanicPrecisionBackend;

        assert_eq!(
            // SAFETY: the prepared/base descriptors are live and the empty result
            // has no non-null output spans; profile validation rejects pre-dispatch.
            unsafe {
                fp32_prepared.execute_precision_f64(&mut backend, &fp32_matrix, &mut f64_result)
            },
            Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity"
            ))
        );

        config.precision = PrecisionProfile::Mixed;
        let mixed_prepared = prepare_continuous_execution(&config, 8, 2).unwrap();
        let fp64_matrix =
            MatrixHandle::host_with_precision(GAFIME_BACKEND_CPU, PrecisionProfile::Fp64, 8, 2);
        assert_eq!(
            // SAFETY: the prepared/base descriptors are live and the empty result
            // has no non-null output spans; profile validation rejects pre-dispatch.
            unsafe {
                mixed_prepared.execute_precision_f64(&mut backend, &fp64_matrix, &mut f64_result)
            },
            Err(OrchestratorError::InvalidPlan(
                "prepared execution precision does not match resident matrix identity"
            ))
        );
    }

    #[test]
    fn prepared_execution_requests_immutable_protocol_without_mutating_plan() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;

        let prepared = prepare_continuous_execution(&config, 8, 2).unwrap();
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 2);
        let mut result = GafimeResultTable::default();
        let mut backend = RecordingBackend::default();

        // SAFETY: `prepared` owns every protocol span; the recording backend
        // deliberately does not dereference this zero-capacity result table.
        unsafe { prepared.execute(&mut backend, &matrix, &mut result) }.unwrap();

        assert_ne!(
            backend.launch_flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );
        let first_generation = backend.descriptor_generation;
        assert_ne!(first_generation, 0);
        // SAFETY: the same prepared owner remains live and the recording
        // backend does not dereference the zero-capacity result table.
        unsafe { prepared.execute(&mut backend, &matrix, &mut result) }.unwrap();
        assert_eq!(backend.descriptor_generation, first_generation);
        assert_eq!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );
        assert_eq!(
            prepared.plan().protocol().reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT],
            0
        );

        let second = prepare_continuous_execution(&config, 8, 2).unwrap();
        // SAFETY: `second` owns its protocol graph and the recording backend
        // does not dereference the zero-capacity result table.
        unsafe { second.execute(&mut backend, &matrix, &mut result) }.unwrap();
        assert_ne!(backend.descriptor_generation, first_generation);
    }

    #[test]
    fn generated_batches_preserve_order_and_result_metadata_without_cache_keys() {
        let plan = generated_pair_plan(GafimeRankSpec {
            top_k: 6,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        plan.validate().unwrap();
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
        let mut result = TestResultTable::new(6, 2, 1);
        let mut backend = RecordingBackend::default();
        let mut backend_context = 0u8;
        let backend_context_ptr = (&mut backend_context as *mut u8).cast();
        result.raw.flags = 0x80;
        result.raw.backend_private = backend_context_ptr;
        result.raw.reserved = [9; 8];

        let stats = execute_compiled_plan_with_batch_words(
            &plan,
            Some(77),
            4,
            &mut backend,
            &matrix,
            &mut result.raw,
        )
        .unwrap();

        assert_eq!(stats.launched_chunks, 3);
        assert_eq!(stats.graph_replays, 0);
        assert_eq!(stats.rows_written, 6);
        assert_eq!(result.raw.row_count, 6);
        assert_eq!(result.candidate_ids, vec![1, 5, 2, 3, 0, 4]);
        assert_eq!(result.ranks, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(
            result.combo_indices,
            vec![4, 5, 5, 2, 4, 2, 0, 5, 4, 0, 0, 2]
        );
        assert_eq!(result.raw.flags, 0x80);
        assert_eq!(result.raw.backend_private, backend_context_ptr);
        assert_eq!(result.raw.reserved, [9; 8]);
        assert_eq!(plan.materialized_descriptor_words(), 0);
        assert!(backend.launches.iter().all(|launch| {
            launch.flags & GAFIME_LAUNCH_FLAG_GRAPH == 0
                && launch.flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL == 0
                && launch.descriptor_generation == 0
                && launch.combo_indices.len() <= 4
                && launch.result_flags == 0x80
                && launch.result_backend_private == backend_context_ptr as usize
                && launch.result_reserved == [9; 8]
        }));
        assert_eq!(
            backend
                .launches
                .iter()
                .flat_map(|launch| launch.combo_indices.iter().copied())
                .collect::<Vec<_>>(),
            vec![4, 0, 4, 5, 4, 2, 0, 5, 0, 2, 5, 2]
        );
    }

    #[test]
    fn generated_mixed_arity_batches_preserve_order_and_use_effective_local_k() {
        let rank = GafimeRankSpec {
            top_k: u32::MAX,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        };
        let plan = generated_mixed_arity_plan(rank);
        plan.validate().unwrap();
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 5);
        let mut result = TestResultTable::new(9, 2, 1);
        let mut backend = RecordingBackend::default();

        let stats = execute_compiled_plan_with_batch_words(
            &plan,
            None,
            4,
            &mut backend,
            &matrix,
            &mut result.raw,
        )
        .unwrap();

        assert_eq!(stats.launched_chunks, 4);
        assert_eq!(stats.rows_written, 9);
        assert_eq!(result.raw.row_count, 9);
        assert_eq!(result.candidate_ids, vec![4, 5, 2, 3, 8, 0, 6, 1, 7]);
        assert_eq!(
            result.metric_values,
            vec![7.0, 5.0, 4.0, 4.0, 4.0, 3.0, 3.0, 1.0, 1.0]
        );
        assert_eq!(
            result.combo_indices,
            vec![
                4,
                3,
                4,
                1,
                4,
                u32::MAX,
                4,
                0,
                3,
                1,
                3,
                u32::MAX,
                0,
                3,
                1,
                u32::MAX,
                0,
                1,
            ]
        );
        assert_eq!(
            backend
                .launches
                .iter()
                .map(|launch| (launch.rank_top_k, launch.result_capacity))
                .collect::<Vec<_>>(),
            vec![(3, 3), (2, 2), (2, 2), (2, 2)]
        );
        assert_eq!(
            backend
                .launches
                .iter()
                .flat_map(|launch| launch.combo_indices.iter().copied())
                .collect::<Vec<_>>(),
            vec![3, 1, 4, 4, 0, 4, 3, 4, 1, 0, 3, 0, 1, 3, 1]
        );
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn generated_top_k_merges_bounded_batch_rows_with_global_candidate_ids() {
        let cases = [
            (1, [1, 5, 2], [4, 5, 5, 2, 4, 2], [9.0, 7.0, 6.0]),
            (0, [4, 0, 3], [0, 2, 4, 0, 0, 5], [2.0, 4.0, 5.0]),
        ];
        for (descending, candidate_ids, combos, scores) in cases {
            let plan = generated_pair_plan(GafimeRankSpec {
                top_k: 3,
                primary_metric: GAFIME_METRIC_PEARSON,
                descending,
                include_ties: 0,
                reserved: [0; 4],
            });
            plan.validate().unwrap();
            let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
            let mut result = TestResultTable::new(3, 2, 1);
            let mut backend = RecordingBackend::default();

            let stats = execute_compiled_plan_with_batch_words(
                &plan,
                Some(91),
                4,
                &mut backend,
                &matrix,
                &mut result.raw,
            )
            .unwrap();

            assert_eq!(stats.launched_chunks, 3);
            assert_eq!(stats.rows_written, 3);
            assert_eq!(result.raw.row_count, 3);
            assert_eq!(&result.candidate_ids[..3], &candidate_ids);
            assert_eq!(&result.ranks[..3], &[0, 1, 2]);
            assert_eq!(&result.combo_indices[..6], &combos);
            assert_eq!(&result.metric_values[..3], &scores);
            assert_eq!(plan.materialized_descriptor_words(), 0);
            assert!(backend.launches.iter().all(|launch| {
                launch.result_capacity <= 3
                    && launch.flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL == 0
                    && launch.descriptor_generation == 0
            }));
        }
    }

    #[test]
    fn streamed_rank_merge_breaks_equal_scores_by_global_candidate_id() {
        let make_row = |candidate_id| StreamedRankedRow {
            score: 1.0,
            candidate_id,
            combo: Vec::new(),
            metrics: vec![1.0],
            family: GAFIME_FAMILY_CONTINUOUS,
            row_flags: 0,
        };
        let mut rows = Vec::with_capacity(2);
        for candidate_id in [9, 2, 5] {
            consider_ranked_row(&mut rows, make_row(candidate_id), 2, true);
        }

        assert_eq!(
            rows.iter().map(|row| row.candidate_id).collect::<Vec<_>>(),
            vec![2, 5]
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.capacity(), 2);
    }

    #[test]
    fn prepared_rank_override_streams_complete_family_and_clears_permutations() {
        let plan = generated_pair_plan(GafimeRankSpec {
            top_k: 1,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        })
        .with_permutations(GafimePermutationSchedule {
            permutation_count: 11,
            seed: 7,
            ..Default::default()
        });
        let prepared = prepare_generated_pair_plan(plan);
        let rank = GafimeRankSpec {
            top_k: 2,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 0,
            include_ties: 0,
            reserved: [0; 4],
        };
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
        let mut result = TestResultTable::new(2, 2, 1);
        let mut backend = RecordingBackend::default();

        assert_eq!(prepared.ranked_result_capacity(rank).unwrap(), 2);
        // SAFETY: `prepared` owns every input descriptor and `result` owns all
        // output buffers described by its rebound raw table.
        unsafe { prepared.execute_ranked(rank, &mut backend, &matrix, &mut result.raw) }.unwrap();

        assert_eq!(result.raw.row_count, 2);
        assert_eq!(&result.candidate_ids[..2], &[4, 0]);
        assert_eq!(&result.combo_indices[..4], &[0, 2, 4, 0]);
        assert!(backend
            .launches
            .iter()
            .all(|launch| launch.permutation_count == 0 && launch.result_capacity <= 2));
        assert_eq!(prepared.plan().permutations().permutation_count, 11);
        assert_eq!(prepared.plan().materialized_descriptor_words(), 0);
    }

    #[test]
    fn launch_protocol_materializes_a_valid_plan_just_above_generation_threshold() {
        let higher_features = (0..1_025).collect::<Vec<_>>();
        let rank = GafimeRankSpec {
            top_k: 1,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        };
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = u64::MAX;
        let prepared = prepare_ranked_continuous_execution_for_feature_orders(
            &config,
            8,
            1_025,
            &[],
            &higher_features,
            false,
            false,
            rank,
        )
        .unwrap();

        assert!(prepared.plan().uses_generated_descriptors());
        assert_eq!(prepared.plan().logical_descriptor_words(), 1_049_600);
        assert!(prepared.plan().logical_descriptor_words() > DEFAULT_DESCRIPTOR_BATCH_WORDS as u64);
        assert_eq!(prepared.plan().materialized_descriptor_words(), 0);

        let checked = prepared.try_launch_protocol().unwrap();
        let protocol = prepared.launch_protocol();
        assert_eq!(checked.combo_indices.ptr, protocol.combo_indices.ptr);
        assert_eq!(protocol.combo_indices.len, 1_049_600);
        assert!(!protocol.combo_indices.ptr.is_null());
        assert_eq!(protocol.chunk_count, 1);
        assert_ne!(protocol.flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, 0);
        assert_ne!(protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT], 0);
        // SAFETY: launch_protocol materialized and owns exactly the declared
        // combo-index words; `prepared` remains live through this assertion.
        let descriptors = unsafe {
            core::slice::from_raw_parts(
                protocol.combo_indices.ptr,
                protocol.combo_indices.len as usize,
            )
        };
        assert_eq!(&descriptors[..4], &[0, 1, 0, 2]);
        assert_eq!(&descriptors[descriptors.len() - 2..], &[1_023, 1_024]);
        assert_eq!(prepared.plan().materialized_descriptor_words(), 1_049_600);
        prepared.plan().validate().unwrap();
    }

    #[test]
    fn generated_streaming_rejects_unbounded_tie_expansion_before_launch() {
        let plan = generated_pair_plan(GafimeRankSpec {
            top_k: 3,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 1,
            reserved: [0; 4],
        });
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
        let mut result = TestResultTable::new(3, 2, 1);
        let mut backend = RecordingBackend::default();

        let error = execute_compiled_plan_with_batch_words(
            &plan,
            None,
            4,
            &mut backend,
            &matrix,
            &mut result.raw,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OrchestratorError::Unsupported("rank.include_ties is unsupported")
        );
        assert!(backend.launches.is_empty());
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn generated_ranked_graph_flag_is_rejected_before_backend_launch() {
        let plan = generated_pair_plan(GafimeRankSpec {
            top_k: 3,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        })
        .with_flags(GAFIME_LAUNCH_FLAG_GRAPH);
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
        let mut result = TestResultTable::new(3, 2, 1);
        let mut backend = RecordingBackend::default();

        let error = execute_compiled_plan_with_batch_words(
            &plan,
            None,
            4,
            &mut backend,
            &matrix,
            &mut result.raw,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OrchestratorError::Unsupported(
                "generated ranked continuous execution does not support graph capture"
            )
        );
        assert!(backend.launches.is_empty());
        assert_eq!(result.raw.row_count, 0);
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn generated_ranked_public_preparation_rejects_graph_request() {
        let higher_features = (0..1_025).collect::<Vec<_>>();
        let rank = GafimeRankSpec {
            top_k: 1,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        };
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            graph_requested: true,
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = u64::MAX;

        for backend_kind in [
            GAFIME_BACKEND_CUDA,
            GAFIME_BACKEND_ROCM,
            GAFIME_BACKEND_METAL,
        ] {
            config.backend_kind = backend_kind;
            let error = prepare_ranked_continuous_execution_for_feature_orders(
                &config,
                8,
                1_025,
                &[],
                &higher_features,
                false,
                false,
                rank,
            )
            .unwrap_err();

            assert_eq!(
                error,
                OrchestratorError::Unsupported(
                    "generated ranked continuous execution does not support graph capture"
                )
            );
        }
    }

    #[test]
    fn generated_streaming_rejects_backend_private_replacement() {
        let plan = generated_pair_plan(GafimeRankSpec {
            top_k: 3,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        });
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CPU, 8, 6);
        let mut result = TestResultTable::new(3, 2, 1);
        let mut backend = RecordingBackend {
            replace_backend_private: true,
            ..Default::default()
        };

        let error = execute_compiled_plan_with_batch_words(
            &plan,
            None,
            4,
            &mut backend,
            &matrix,
            &mut result.raw,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OrchestratorError::Unsupported("streamed backend changed result backend_private")
        );
        assert_eq!(plan.materialized_descriptor_words(), 0);
    }

    #[test]
    fn hundred_million_ranked_public_preparation_is_bounded_and_storage_aware() {
        let higher_features = (0..20_000).collect::<Vec<_>>();
        let rank = GafimeRankSpec {
            top_k: 32,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        };
        let mut config = EngineConfig {
            precision: PrecisionProfile::Fp32,
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 100_000_000;
        config.budget.vram_budget_mb = 10;

        let prepared = prepare_ranked_continuous_execution_for_feature_orders(
            &config,
            32,
            20_000,
            &[],
            &higher_features,
            false,
            false,
            rank,
        )
        .unwrap();

        assert_eq!(prepared.plan().planned_row_count(), 100_000_000);
        assert_eq!(prepared.plan().logical_descriptor_words(), 200_000_000);
        assert_eq!(
            prepared
                .plan()
                .peak_descriptor_batch_words(DEFAULT_DESCRIPTOR_BATCH_WORDS),
            1_048_576
        );
        assert_eq!(prepared.result_capacity(), 32);
        assert_eq!(prepared.ranked_result_capacity(rank).unwrap(), 32);
        assert_eq!(prepared.plan().materialized_descriptor_words(), 0);
        assert_eq!(
            continuous_plan_device_footprint_bytes(
                32,
                20_000,
                prepared.plan(),
                PrecisionProfile::Fp32,
            ),
            10_460_700
        );
        assert_eq!(
            continuous_device_footprint_bytes(32, 20_000, 1, 100_000_000, 200_000_000,),
            1_203_120_412
        );
        let oversized_rank = GafimeRankSpec {
            top_k: 100_000,
            ..rank
        };
        assert!(
            continuous_plan_device_footprint_bytes_for_rank(
                32,
                20_000,
                prepared.plan(),
                PrecisionProfile::Fp32,
                oversized_rank,
            ) > 10 * 1024 * 1024
        );
        assert_eq!(
            prepared.ranked_result_capacity(oversized_rank),
            Err(OrchestratorError::Unsupported(
                "rank override device footprint exceeds budget.vram_budget_mb"
            ))
        );
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CUDA, 32, 20_000);
        let mut result = GafimeResultTable::default();
        let mut backend = RecordingBackend::default();
        assert_eq!(
            // SAFETY: `prepared` owns every protocol span and the empty result has
            // no non-null spans; footprint validation rejects before execution.
            unsafe { prepared.execute_ranked(oversized_rank, &mut backend, &matrix, &mut result) },
            Err(OrchestratorError::Unsupported(
                "rank override device footprint exceeds budget.vram_budget_mb"
            ))
        );
        assert!(backend.launches.is_empty());
        assert_eq!(
            prepared.try_launch_protocol(),
            Err(OrchestratorError::Unsupported(
                "generated launch protocol materialization exceeds the compatibility host-memory budget"
            ))
        );
        assert_eq!(prepared.plan().materialized_descriptor_words(), 0);

        config.budget.vram_budget_mb = 9;
        assert_eq!(
            prepare_ranked_continuous_execution_for_feature_orders(
                &config,
                32,
                20_000,
                &[],
                &higher_features,
                false,
                false,
                rank,
            )
            .unwrap_err(),
            OrchestratorError::Unsupported(
                "continuous plan device footprint exceeds budget.vram_budget_mb"
            )
        );
    }

    #[test]
    fn gafime_py_unranked_preparation_rejects_hundred_million_rows_before_result_allocation() {
        let higher_features = (0..20_000).collect::<Vec<_>>();
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 100_000_000;

        let error = prepare_continuous_execution_for_feature_orders(
            &config,
            2,
            20_000,
            &[],
            &higher_features,
            false,
            false,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OrchestratorError::Unsupported(
                "unranked continuous candidate storage exceeds the host-memory budget"
            )
        );
    }

    #[test]
    fn explicit_cuda_config_stays_cuda_without_cpu_fallback() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 1_000;

        let prepared = prepare_continuous_execution(&config, 32, 4).unwrap();

        assert_eq!(prepared.plan().protocol().backend_kind, GAFIME_BACKEND_CUDA);
        assert_eq!(
            prepared.plan().protocol().permutations.permutation_count,
            config.permutation_tests
        );
        assert_eq!(
            prepared.plan().protocol().permutations.seed,
            config.random_seed
        );
        assert_eq!(prepared.result_capacity(), 10);
    }

    #[test]
    fn explicit_metal_config_stays_metal_without_cpu_fallback() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_METAL,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 1_000;

        let prepared = prepare_continuous_execution(&config, 32, 4).unwrap();

        assert_eq!(
            prepared.plan().protocol().backend_kind,
            GAFIME_BACKEND_METAL
        );
        assert_eq!(prepared.result_capacity(), 10);
    }

    #[test]
    fn mi_approximate_sets_cpu_launch_flag() {
        let mut config = EngineConfig {
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = 1_000;
        config.mi_approximate = true;

        let prepared = prepare_continuous_execution(&config, 32, 4).unwrap();

        assert_ne!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_MI_APPROX,
            0
        );
    }

    #[test]
    fn graph_request_reaches_plan_and_schedule() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = 1_000;
        config.graph_requested = true;

        let prepared = prepare_continuous_execution(&config, 32, 4).unwrap();

        assert_ne!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_GRAPH,
            0
        );
        assert!(prepared.schedule().decision().graph_requested);
    }

    #[test]
    fn device_footprint_sums_buffers_and_saturates() {
        // 100 rows, 4 cols, 2 metrics, 10 planned rows, 20 combo slots.
        // Includes the CUDA fp32 24-byte target/unary statistic records and
        // the bounded target-rank cache attempted for every short matrix.
        let bytes = continuous_device_footprint_bytes(100, 4, 2, 10, 20);
        assert_eq!(bytes, 1600 + 400 + 16 + 24 + 96 + 800 + 80 + 8 + 80);
        // Ranked buffers add effective-K indices/gather values and bounded
        // partial score/index scratch without replacing batch-wide metrics.
        assert_eq!(
            continuous_launch_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp32,
                1_000,
                2_000,
                2,
                32,
                0,
            ),
            8_000 + 8_000 + 128 + 256 + 2_048
        );
        // Huge inputs saturate instead of overflowing.
        assert_eq!(
            continuous_device_footprint_bytes(u64::MAX, u32::MAX, u64::MAX, u64::MAX, u64::MAX),
            u64::MAX
        );
    }

    #[test]
    fn metal_device_footprint_uses_native_matrix_and_launch_shapes() {
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp32,
                100,
                4,
            ),
            1_600 + 400 + 16 + 24 + 96 + 800
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_METAL,
                PrecisionProfile::Fp32,
                100,
                4,
            ),
            1_600 + 400 + 16 + 400
        );
        assert_eq!(topk_partial_block_count(GAFIME_BACKEND_CUDA, 1_000, 32), 8);
        assert_eq!(topk_partial_block_count(GAFIME_BACKEND_CUDA, 257, 100), 3);
        assert_eq!(topk_partial_block_count(GAFIME_BACKEND_ROCM, 257, 100), 2);
        assert_eq!(
            topk_partial_block_count(GAFIME_BACKEND_METAL, 1_000, 32),
            16
        );
        assert_eq!(
            continuous_launch_device_footprint_bytes(
                GAFIME_BACKEND_METAL,
                PrecisionProfile::Fp32,
                1_000,
                2_000,
                2,
                32,
                3,
            ),
            8_000 + 8_000 + 128 + 256 + 4_096 + 3 * 40 + 24 + 24
        );

        let plan = CompiledPlan::single_chunk(
            GAFIME_BACKEND_METAL,
            100,
            4,
            GAFIME_FAMILY_CONTINUOUS,
            1,
            vec![0, 1, 2, 3],
            vec![GAFIME_METRIC_PEARSON],
        );
        assert_eq!(
            continuous_plan_device_footprint_bytes(100, 4, &plan, PrecisionProfile::Fp32),
            2_416 + 4 + 16 + 16 + 40 + 24
        );
    }

    #[test]
    fn typed_matrix_forecast_matches_backend_profile_layouts() {
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp32,
                100,
                4,
            ),
            2_936
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Mixed,
                100,
                4,
            ),
            2_976
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp64,
                100,
                4,
            ),
            4_992
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp32,
                4_096,
                4,
            ),
            4_096 * 4 * 4 + 4_096 * 4 + 4 * 4 + 24 + 4 * 24 + 4_096 * 8
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_CUDA,
                PrecisionProfile::Fp32,
                4_097,
                4,
            ),
            4_097 * 4 * 4 + 4_097 * 4 + 4 * 4 + 24 + 4 * 24
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_ROCM,
                PrecisionProfile::Fp32,
                100,
                4,
            ),
            2_096
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_ROCM,
                PrecisionProfile::Mixed,
                100,
                4,
            ),
            2_152
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(
                GAFIME_BACKEND_ROCM,
                PrecisionProfile::Fp64,
                100,
                4,
            ),
            4_152
        );
    }

    #[test]
    fn staged_forecast_retains_each_native_buffer_high_water() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            precision: PrecisionProfile::Fp32,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 5;
        config.budget.max_combinations_per_k = u64::MAX;
        config.budget.vram_budget_mb = 0;

        let unary_features = (0..100).collect::<Vec<_>>();
        let higher_features = (0..6).collect::<Vec<_>>();
        let unary = prepare_continuous_execution_for_feature_orders(
            &config,
            32,
            100,
            &unary_features,
            &[],
            true,
            false,
        )
        .unwrap();
        let higher = prepare_continuous_execution_for_feature_orders(
            &config,
            32,
            100,
            &[],
            &higher_features,
            false,
            false,
        )
        .unwrap();
        assert_eq!(unary.plan().planned_row_count(), 100);
        assert_eq!(unary.plan().logical_descriptor_words(), 100);
        assert_eq!(higher.plan().planned_row_count(), 56);
        assert_eq!(higher.plan().logical_descriptor_words(), 180);

        let matrix_bytes = continuous_matrix_device_footprint_bytes(
            GAFIME_BACKEND_CUDA,
            PrecisionProfile::Fp32,
            32,
            100,
        );
        let staged = continuous_staged_device_footprint_bytes(&[&unary, &higher]);
        // Unary owns 100 descriptor words. The following 180-word request grows
        // that retained native capacity geometrically to 200 words, and the
        // allocator temporarily owns both the 100- and 200-word allocations.
        assert_eq!(staged, matrix_bytes + 100 * 4 + 4 + 100 * 4 + 200 * 4);
        assert!(
            staged
                > continuous_plan_device_footprint_bytes(
                    32,
                    100,
                    unary.plan(),
                    PrecisionProfile::Fp32,
                )
        );
        assert!(
            staged
                > continuous_plan_device_footprint_bytes(
                    32,
                    100,
                    higher.plan(),
                    PrecisionProfile::Fp32,
                )
        );
    }

    #[test]
    fn metal_sequence_keeps_runtime_scratch_execution_local() {
        let first = LaunchFootprintShape {
            launch_rows: 1_000,
            descriptor_words: 1_000,
            metric_count: 1,
            chunk_count: 1,
            descriptor_generation: 10,
            ..Default::default()
        };
        let second = LaunchFootprintShape {
            launch_rows: 1,
            descriptor_words: 2_000,
            metric_count: 1,
            chunk_count: 1,
            descriptor_generation: 11,
            ..Default::default()
        };

        // First: 4,068 descriptor + 4,000 runtime bytes. Second: replace the
        // old 4,068-byte cache with an 8,068-byte cache, then use 4 runtime
        // bytes. Runtime buffers from the first launch are not retained.
        assert_eq!(
            metal_launch_sequence_peak_bytes(PrecisionProfile::Fp32, 0, [first, second],),
            4_068 + 8_068,
        );
    }

    #[test]
    fn retained_allocator_models_paired_growth_and_generated_batch_order() {
        let mut pair_simulation = DeviceMemoryPeakSimulation::new(100);
        let mut descriptor_capacity = 10;
        let mut metric_id_capacity = 5;
        simulate_buffer_pair_growth(
            &mut pair_simulation,
            &mut descriptor_capacity,
            15,
            4,
            &mut metric_id_capacity,
            8,
            4,
        );
        assert_eq!((descriptor_capacity, metric_id_capacity), (20, 10));
        // The native paired reservation observes both replacements while the
        // 100-byte resident set still owns both old allocations.
        assert_eq!(pair_simulation.peak_bytes, 100 + 20 * 4 + 10 * 4);

        let higher_features = (0..1_001).collect::<Vec<_>>();
        let source = crate::plan::combos::CombinationDescriptorSource::new(&[], &higher_features);
        let rank = GafimeRankSpec {
            top_k: 32,
            primary_metric: GAFIME_METRIC_PEARSON,
            descending: 1,
            include_ties: 0,
            reserved: [0; 4],
        };
        let chunks = vec![
            GafimeArityChunk {
                arity: 2,
                family: GAFIME_FAMILY_CONTINUOUS,
                shape_hint_index: 0,
                combo_count: 500_500,
                descriptor_count: 500_500,
                ..Default::default()
            },
            GafimeArityChunk {
                arity: 3,
                family: GAFIME_FAMILY_CONTINUOUS,
                shape_hint_index: 1,
                combo_row_offset: 500_500,
                combo_count: 400_000,
                local_chunk_id: 1,
                descriptor_offset: 1_001_000,
                descriptor_count: 400_000,
                ..Default::default()
            },
        ];
        let mut pair_shape = crate::plan::shapes::default_shape_hint(GAFIME_BACKEND_CUDA, 2);
        pair_shape.vendor_hint = 2;
        let mut triple_shape = crate::plan::shapes::default_shape_hint(GAFIME_BACKEND_CUDA, 3);
        triple_shape.vendor_hint = 2;
        let plan = CompiledPlan::from_combination_parts(
            GAFIME_BACKEND_CUDA,
            32,
            1_001,
            3,
            source,
            vec![GAFIME_METRIC_PEARSON],
            chunks,
            vec![pair_shape, triple_shape],
            rank,
            GafimePermutationSchedule::default(),
        );
        plan.validate().unwrap();
        let shapes = generated_ranked_launch_footprint_shapes(&plan, rank);
        assert_eq!(shapes.len(), 2);
        assert_eq!(
            (shapes[0].launch_rows, shapes[0].descriptor_words),
            (500_500, 1_001_000),
        );
        // The arity-three family has two batches. Its first/max batch alone is
        // sufficient because the remaining 50,475-row batch is strictly smaller.
        assert_eq!(
            (shapes[1].launch_rows, shapes[1].descriptor_words),
            (349_525, 1_048_575),
        );
        // The second descriptor request grows the retained 1,001,000-word
        // capacity to 2,002,000 words while the first allocation is still live.
        assert_eq!(
            retained_launch_sequence_peak_bytes(PrecisionProfile::Fp32, 0, shapes),
            15_015_476,
        );
    }

    #[test]
    fn native_allocation_simulators_saturate_on_unrepresentable_shapes() {
        let shape = LaunchFootprintShape {
            launch_rows: u64::MAX,
            descriptor_words: u64::MAX,
            metric_count: 2,
            chunk_count: u64::MAX,
            descriptor_generation: 1,
            ..Default::default()
        };
        assert_eq!(
            retained_launch_sequence_peak_bytes(PrecisionProfile::Fp64, 0, [shape]),
            u64::MAX,
        );
        assert_eq!(
            metal_launch_sequence_peak_bytes(PrecisionProfile::Fp32, 0, [shape]),
            u64::MAX,
        );
    }

    #[test]
    fn vram_budget_rejects_oversized_gpu_plan() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 200_000;
        config.budget.vram_budget_mb = 1;
        // 512 features -> C(512,2) = 130,816 pair combos -> the metric-value buffer
        // alone (~1.05 MB) exceeds the 1 MB budget.
        assert!(matches!(
            prepare_continuous_execution(&config, 32, 512),
            Err(OrchestratorError::Unsupported(_))
        ));
    }

    #[test]
    fn vram_boundary_allows_fp32_but_rejects_mixed_and_fp64() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 200_000;
        config.budget.vram_budget_mb = 3;

        config.precision = PrecisionProfile::Fp32;
        let fp32 = prepare_continuous_execution(&config, 32, 512).unwrap();
        assert_eq!(fp32.precision(), PrecisionProfile::Fp32);

        for precision in [PrecisionProfile::Mixed, PrecisionProfile::Fp64] {
            config.precision = precision;
            assert_eq!(
                prepare_continuous_execution(&config, 32, 512).unwrap_err(),
                OrchestratorError::Unsupported(
                    "continuous plan device footprint exceeds budget.vram_budget_mb"
                ),
                "precision={precision:?}"
            );
        }
    }

    #[test]
    fn vram_budget_allows_normal_gpu_plan() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2],
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 1_000;
        // Default vram_budget_mb (6144) easily fits a small plan.
        assert!(prepare_continuous_execution(&config, 32, 8).is_ok());
    }

    #[test]
    fn native_transition_peak_is_checked_before_launch() {
        let mut config = EngineConfig {
            backend_kind: GAFIME_BACKEND_CUDA,
            metric_ids: vec![GAFIME_METRIC_PEARSON],
            permutation_tests: 0,
            num_repeats: 1,
            ..EngineConfig::default()
        };
        config.budget.max_comb_size = 1;
        config.budget.max_combinations_per_k = 8;
        config.budget.vram_budget_mb = 1;
        let prepared = prepare_continuous_execution(&config, 32, 4).unwrap();
        let matrix = MatrixHandle::host(GAFIME_BACKEND_CUDA, 32, 4);
        let mut result = GafimeResultTable::default();
        let mut backend = RecordingBackend {
            device_memory_peak_bytes: Some(2 * 1024 * 1024),
            ..RecordingBackend::default()
        };

        assert_eq!(
            // SAFETY: `prepared` owns every protocol span and the empty result has
            // no non-null spans; memory preflight rejects before execution.
            unsafe { prepared.execute(&mut backend, &matrix, &mut result) },
            Err(OrchestratorError::Unsupported(
                "continuous execution device-memory peak exceeds budget.vram_budget_mb"
            ))
        );
        assert_eq!(backend.device_memory_preflights, 1);
        assert!(backend.launches.is_empty());
    }
}
