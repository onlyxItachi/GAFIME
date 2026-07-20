use gafime_types::{
    BackendKind, GafimeLaunchProtocol, GafimePermutationSchedule, GafimeRankSpec,
    GafimeResultTable, GafimeSliceU32, GAFIME_ABI_VERSION, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA,
    GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_LAUNCH_FLAG_GRAPH,
    GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, GAFIME_LAUNCH_FLAG_MI_APPROX,
    GAFIME_RESULT_FLAG_GRAPH_REPLAYED,
};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    backend::{BackendExecutionStats, ComputeBackend, MatrixHandle},
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
    pub fn execute<B: ComputeBackend>(
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

    /// Score the complete prepared family with a bounded rank override. This is
    /// the generated-plan path for host-orchestrated maxT extrema: descriptors
    /// are streamed, only K rows are retained, and the plan's permutation
    /// schedule is disabled because the caller supplies the target being scored.
    pub fn execute_ranked<B: ComputeBackend>(
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

    fn validate_rank_device_budget(&self, rank: GafimeRankSpec) -> OrchestratorResult<()> {
        let Some(budget_bytes) = self.device_budget_bytes else {
            return Ok(());
        };
        let footprint =
            continuous_plan_device_footprint_bytes_for_rank(self.rows, self.cols, &self.plan, rank);
        if footprint > budget_bytes {
            return Err(OrchestratorError::Unsupported(
                "rank override device footprint exceeds budget.vram_budget_mb",
            ));
        }
        Ok(())
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
        return backend.execute(matrix, &protocol, result);
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
        let stats = backend.execute(matrix, &protocol, &mut batch_result.raw)?;
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

fn validate_execution_device_budget<B: ComputeBackend>(
    backend: &mut B,
    matrix: &MatrixHandle,
    protocol: &GafimeLaunchProtocol,
    device_budget_bytes: Option<u64>,
) -> OrchestratorResult<()> {
    let Some(device_budget_bytes) = device_budget_bytes else {
        return Ok(());
    };
    if backend
        .execution_device_memory_peak_bytes(matrix, protocol)?
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
struct StreamedRankedRow {
    score: f32,
    candidate_id: u64,
    combo: Vec<u32>,
    metrics: Vec<f32>,
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
        let footprint = continuous_plan_device_footprint_bytes(rows, cols, &plan);
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
        descriptor_generation: next_descriptor_generation(),
        rows,
        cols,
        device_budget_bytes,
    })
}

/// Estimate peak resident device buffers for this plan. Generated ranked plans
/// retain the maximum capacity reached across descriptor batches; metric values
/// therefore scale with batch rows, while result/gather buffers scale with K.
pub fn continuous_plan_device_footprint_bytes(rows: u64, cols: u32, plan: &CompiledPlan) -> u64 {
    continuous_plan_device_footprint_bytes_for_rank(rows, cols, plan, plan.rank())
}

fn continuous_plan_device_footprint_bytes_for_rank(
    rows: u64,
    cols: u32,
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
) -> u64 {
    let generated_ranked = plan.uses_generated_descriptors() && rank.top_k > 0;
    let launch_buffers = if generated_ranked {
        generated_ranked_launch_device_footprint_bytes(plan, rank)
    } else {
        continuous_launch_device_footprint_bytes(
            plan.backend_kind(),
            plan.planned_row_count(),
            plan.logical_descriptor_words(),
            u64::from(plan.metric_count()),
            u64::from(rank.top_k),
            plan.chunks().len() as u64,
        )
    };
    continuous_matrix_device_footprint_bytes(plan.backend_kind(), rows, cols)
        .saturating_add(u64::from(plan.metric_count()).saturating_mul(4))
        .saturating_add(launch_buffers)
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
    continuous_matrix_device_footprint_bytes(GAFIME_BACKEND_CUDA, rows, cols)
        .saturating_add(metric_count.saturating_mul(4))
        .saturating_add(continuous_launch_device_footprint_bytes(
            GAFIME_BACKEND_CUDA,
            planned_rows,
            combo_slots,
            metric_count,
            0,
            0,
        ))
}

fn continuous_matrix_device_footprint_bytes(
    backend_kind: BackendKind,
    rows: u64,
    cols: u32,
) -> u64 {
    const F32_BYTES: u64 = 4;
    const TARGET_STATS_BYTES: u64 = 16;
    const FEATURE_STATS_BYTES: u64 = 16;

    let base = rows
        .saturating_mul(u64::from(cols))
        .saturating_mul(F32_BYTES)
        .saturating_add(rows.saturating_mul(F32_BYTES))
        .saturating_add(u64::from(cols).saturating_mul(F32_BYTES));
    if backend_kind == GAFIME_BACKEND_METAL {
        base
    } else {
        base.saturating_add(TARGET_STATS_BYTES)
            .saturating_add(u64::from(cols).saturating_mul(FEATURE_STATS_BYTES))
    }
}

fn generated_ranked_launch_device_footprint_bytes(
    plan: &CompiledPlan,
    rank: GafimeRankSpec,
) -> u64 {
    let effective_top_k = effective_ranked_rows(plan, rank);
    let max_words = DEFAULT_DESCRIPTOR_BATCH_WORDS as u64;
    let mut peak_batch_rows = 0u64;
    let mut peak_descriptor_words = 0u64;
    let mut peak_local_top_k = 0u64;
    let mut peak_partial_items = 0u64;

    for chunk in plan.chunks() {
        let arity = u64::from(chunk.arity);
        let batch_rows = chunk.combo_count.min((max_words / arity).max(1));
        let descriptor_words = batch_rows.saturating_mul(arity);
        let local_top_k = batch_rows.min(effective_top_k);
        let partial_items = topk_partial_block_count(plan.backend_kind(), batch_rows, local_top_k)
            .saturating_mul(local_top_k);
        peak_batch_rows = peak_batch_rows.max(batch_rows);
        peak_descriptor_words = peak_descriptor_words.max(descriptor_words);
        peak_local_top_k = peak_local_top_k.max(local_top_k);
        peak_partial_items = peak_partial_items.max(partial_items);
    }

    ranked_launch_device_footprint_bytes(
        plan.backend_kind(),
        peak_batch_rows,
        peak_descriptor_words,
        u64::from(plan.metric_count()),
        peak_local_top_k,
        peak_partial_items,
        u64::from(peak_batch_rows != 0),
    )
}

fn continuous_launch_device_footprint_bytes(
    backend_kind: BackendKind,
    launch_rows: u64,
    descriptor_words: u64,
    metric_count: u64,
    requested_top_k: u64,
    chunk_count: u64,
) -> u64 {
    let effective_top_k = launch_rows.min(requested_top_k);
    let partial_items = topk_partial_block_count(backend_kind, launch_rows, effective_top_k)
        .saturating_mul(effective_top_k);
    ranked_launch_device_footprint_bytes(
        backend_kind,
        launch_rows,
        descriptor_words,
        metric_count,
        effective_top_k,
        partial_items,
        chunk_count,
    )
}

fn ranked_launch_device_footprint_bytes(
    backend_kind: BackendKind,
    launch_rows: u64,
    descriptor_words: u64,
    metric_count: u64,
    effective_top_k: u64,
    partial_items: u64,
    chunk_count: u64,
) -> u64 {
    const F32_BYTES: u64 = 4;
    const U32_BYTES: u64 = 4;

    let descriptor_bytes = descriptor_words.saturating_mul(U32_BYTES);
    let metric_value_bytes = launch_rows
        .saturating_mul(metric_count)
        .saturating_mul(F32_BYTES);
    let selected_index_bytes = effective_top_k.saturating_mul(U32_BYTES);
    let selected_metric_bytes = effective_top_k
        .saturating_mul(metric_count)
        .saturating_mul(F32_BYTES);
    let partial_scratch_bytes = partial_items.saturating_mul(F32_BYTES + U32_BYTES);

    let launch_bytes = descriptor_bytes
        .saturating_add(metric_value_bytes)
        .saturating_add(selected_index_bytes)
        .saturating_add(selected_metric_bytes)
        .saturating_add(partial_scratch_bytes);
    if backend_kind == GAFIME_BACKEND_METAL {
        const METAL_CHUNK_BYTES: u64 = 32;
        const METAL_LAUNCH_INFO_BYTES: u64 = 24;
        const METAL_RANK_INFO_BYTES: u64 = 24;
        launch_bytes
            .saturating_add(chunk_count.saturating_mul(METAL_CHUNK_BYTES))
            .saturating_add(METAL_LAUNCH_INFO_BYTES)
            .saturating_add(u64::from(effective_top_k != 0).saturating_mul(METAL_RANK_INFO_BYTES))
    } else {
        launch_bytes
    }
}

fn topk_partial_block_count(backend_kind: BackendKind, row_count: u64, top_k: u64) -> u64 {
    const CUDA_HIP_TOPK_THREADS_PER_BLOCK: u64 = 256;
    const METAL_TOPK_THREADS_PER_BLOCK: u64 = 64;
    const TOPK_MAX_PARTIAL_BLOCKS: u64 = 4096;

    if row_count == 0 || top_k == 0 {
        return 0;
    }
    let threads_per_block = if backend_kind == GAFIME_BACKEND_METAL {
        METAL_TOPK_THREADS_PER_BLOCK
    } else {
        CUDA_HIP_TOPK_THREADS_PER_BLOCK
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

        fn execution_device_memory_peak_bytes(
            &mut self,
            _matrix: &MatrixHandle,
            _protocol: &GafimeLaunchProtocol,
        ) -> OrchestratorResult<Option<u64>> {
            self.device_memory_preflights += 1;
            Ok(self.device_memory_peak_bytes)
        }

        fn execute(
            &mut self,
            _matrix: &MatrixHandle,
            protocol: &gafime_types::GafimeLaunchProtocol,
            result: &mut GafimeResultTable,
        ) -> OrchestratorResult<BackendExecutionStats> {
            self.launch_flags = protocol.flags;
            self.descriptor_generation = protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT];
            let chunks = unsafe {
                core::slice::from_raw_parts(protocol.chunks, protocol.chunk_count as usize)
            };
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

    struct TestResultTable {
        raw: GafimeResultTable,
        combo_indices: Vec<u32>,
        metric_values: Vec<f32>,
        ranks: Vec<u32>,
        families: Vec<u32>,
        candidate_ids: Vec<u64>,
        row_flags: Vec<u32>,
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

        prepared
            .execute(&mut backend, &matrix, &mut result)
            .unwrap();

        assert_ne!(
            backend.launch_flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );
        let first_generation = backend.descriptor_generation;
        assert_ne!(first_generation, 0);
        prepared
            .execute(&mut backend, &matrix, &mut result)
            .unwrap();
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
        second.execute(&mut backend, &matrix, &mut result).unwrap();
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
        prepared
            .execute_ranked(rank, &mut backend, &matrix, &mut result.raw)
            .unwrap();

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
            continuous_plan_device_footprint_bytes(32, 20_000, prepared.plan()),
            9_776_148
        );
        assert_eq!(
            continuous_device_footprint_bytes(32, 20_000, 1, 100_000_000, 200_000_000,),
            1_202_960_148
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
            prepared.execute_ranked(oversized_rank, &mut backend, &matrix, &mut result),
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
        // Includes 16-byte target stats and 16 bytes of unary stats per column.
        let bytes = continuous_device_footprint_bytes(100, 4, 2, 10, 20);
        assert_eq!(bytes, 1600 + 400 + 16 + 16 + 64 + 80 + 8 + 80);
        // Ranked buffers add effective-K indices/gather values and bounded
        // partial score/index scratch without replacing batch-wide metrics.
        assert_eq!(
            continuous_launch_device_footprint_bytes(GAFIME_BACKEND_CUDA, 1_000, 2_000, 2, 32, 0,),
            8_000 + 8_000 + 128 + 256 + 1_024
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
            continuous_matrix_device_footprint_bytes(GAFIME_BACKEND_CUDA, 100, 4),
            1_600 + 400 + 16 + 16 + 64
        );
        assert_eq!(
            continuous_matrix_device_footprint_bytes(GAFIME_BACKEND_METAL, 100, 4),
            1_600 + 400 + 16
        );
        assert_eq!(topk_partial_block_count(GAFIME_BACKEND_CUDA, 1_000, 32), 4);
        assert_eq!(
            topk_partial_block_count(GAFIME_BACKEND_METAL, 1_000, 32),
            16
        );
        assert_eq!(
            continuous_launch_device_footprint_bytes(GAFIME_BACKEND_METAL, 1_000, 2_000, 2, 32, 3,),
            8_000 + 8_000 + 128 + 256 + 4_096 + 3 * 32 + 24 + 24
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
            continuous_plan_device_footprint_bytes(100, 4, &plan),
            2_016 + 4 + 16 + 16 + 32 + 24
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
            prepared.execute(&mut backend, &matrix, &mut result),
            Err(OrchestratorError::Unsupported(
                "continuous execution device-memory peak exceeds budget.vram_budget_mb"
            ))
        );
        assert_eq!(backend.device_memory_preflights, 1);
        assert!(backend.launches.is_empty());
    }
}
