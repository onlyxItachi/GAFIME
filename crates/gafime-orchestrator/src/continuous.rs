use gafime_types::{
    BackendKind, GafimePermutationSchedule, GafimeRankSpec, GafimeResultTable, GAFIME_BACKEND_CPU,
    GAFIME_BACKEND_CUDA, GAFIME_BACKEND_METAL, GAFIME_BACKEND_ROCM, GAFIME_LAUNCH_FLAG_GRAPH,
    GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL, GAFIME_LAUNCH_FLAG_MI_APPROX,
};

use crate::{
    backend::{BackendExecutionStats, ComputeBackend, MatrixHandle},
    config::EngineConfig,
    plan::{
        combos::{
            build_continuous_plan, build_continuous_plan_for_feature_orders, ContinuousPlanRequest,
        },
        CompiledPlan,
    },
    schedule::ContinuousSchedule,
    OrchestratorError, OrchestratorResult,
};

#[derive(Debug)]
pub struct PreparedContinuousExecution {
    plan: CompiledPlan,
    schedule: ContinuousSchedule,
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

    /// Execute a plan that was validated when this prepared artifact was built.
    /// General callers should use `execute_plan`, which validates arbitrary
    /// plans on every call; compiled artifacts keep this immutable trusted path.
    pub fn execute<B: ComputeBackend>(
        &self,
        backend: &mut B,
        matrix: &MatrixHandle,
        result: &mut GafimeResultTable,
    ) -> OrchestratorResult<BackendExecutionStats> {
        backend.execute(matrix, self.plan.protocol(), result)
    }
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
    let plan = build_continuous_plan(continuous_plan_request(config, rows, cols, backend_kind))?;
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
        continuous_plan_request(config, rows, cols, backend_kind),
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
) -> ContinuousPlanRequest {
    ContinuousPlanRequest {
        backend_kind,
        n_samples: rows,
        n_features: cols,
        max_arity: config.budget.max_comb_size,
        max_combinations_per_arity: config.budget.max_combinations_per_k,
        metric_ids: config.metric_ids.clone(),
        mi_bins: config.mi_bins,
        rank: GafimeRankSpec::default(),
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
    let mut flags = plan.protocol().flags | GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL;
    if config.mi_approximate {
        flags |= GAFIME_LAUNCH_FLAG_MI_APPROX;
    }
    if config.graph_requested {
        flags |= GAFIME_LAUNCH_FLAG_GRAPH;
    }
    if flags != plan.protocol().flags {
        plan = plan.with_flags(flags);
    }
    plan.validate()?;

    // VRAM budget enforcement: fail fast with a clear error instead
    // of OOMing the device when the resident plan would exceed the configured
    // budget. Applies to the GPU backends only (the CPU engine holds no VRAM).
    if matches!(
        backend_kind,
        GAFIME_BACKEND_CUDA | GAFIME_BACKEND_ROCM | GAFIME_BACKEND_METAL
    ) && config.budget.vram_budget_mb > 0
    {
        let planned_rows: u64 = plan.chunks().iter().map(|chunk| chunk.combo_count).sum();
        let combo_slots: u64 = plan
            .chunks()
            .iter()
            .map(|chunk| chunk.combo_count.saturating_mul(chunk.arity as u64))
            .sum();
        let footprint = continuous_device_footprint_bytes(
            rows,
            cols,
            config.metric_ids.len() as u64,
            planned_rows,
            combo_slots,
        );
        let budget_bytes = config.budget.vram_budget_mb.saturating_mul(1024 * 1024);
        if footprint > budget_bytes {
            return Err(OrchestratorError::Unsupported(
                "continuous plan device footprint exceeds budget.vram_budget_mb",
            ));
        }
    }

    let schedule = ContinuousSchedule::for_plan(&plan)?;
    Ok(PreparedContinuousExecution { plan, schedule })
}

/// Estimate the resident device-memory footprint (bytes) of a continuous plan:
/// the feature matrix + target + column means (f32), the combo-index buffer and
/// metric-id buffer (u32), and the metric-value output (f32). Mirrors the buffers
/// the CUDA/ROCm hosts allocate. Saturating to avoid overflow on huge plans.
pub fn continuous_device_footprint_bytes(
    rows: u64,
    cols: u32,
    metric_count: u64,
    planned_rows: u64,
    combo_slots: u64,
) -> u64 {
    const F32: u64 = 4;
    const U32: u64 = 4;
    let features = rows.saturating_mul(cols as u64).saturating_mul(F32);
    let target = rows.saturating_mul(F32);
    let column_means = (cols as u64).saturating_mul(F32);
    let combo_indices = combo_slots.saturating_mul(U32);
    let metric_ids = metric_count.saturating_mul(U32);
    let metric_values = planned_rows
        .saturating_mul(metric_count)
        .saturating_mul(F32);
    features
        .saturating_add(target)
        .saturating_add(column_means)
        .saturating_add(combo_indices)
        .saturating_add(metric_ids)
        .saturating_add(metric_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{
        GAFIME_BACKEND_METAL, GAFIME_LAUNCH_FLAG_GRAPH, GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
        GAFIME_LAUNCH_FLAG_MI_APPROX, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2,
    };

    #[test]
    fn default_config_prepares_cpu_continuous_execution() {
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.budget.max_comb_size = 3;
        config.budget.max_combinations_per_k = 1_000;

        let prepared = prepare_continuous_execution(&config, 32, 5).unwrap();

        assert_eq!(prepared.plan().protocol().backend_kind, GAFIME_BACKEND_CPU);
        assert_eq!(prepared.plan().protocol().permutations.permutation_count, 0);
        assert_eq!(prepared.result_max_arity(), 3);
        assert_eq!(prepared.result_metric_count(), 2);
        assert_eq!(prepared.result_capacity(), 25);
        assert_ne!(
            prepared.plan().protocol().flags & GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL,
            0
        );
    }

    #[test]
    fn explicit_cuda_config_stays_cuda_without_cpu_fallback() {
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
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
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_METAL;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
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
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
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
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON];
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
        // features 100*4*4 + target 100*4 + means 4*4 + combo 20*4 + ids 2*4 + values 10*2*4
        let bytes = continuous_device_footprint_bytes(100, 4, 2, 10, 20);
        assert_eq!(bytes, 1600 + 400 + 16 + 80 + 8 + 80);
        // Huge inputs saturate instead of overflowing.
        assert_eq!(
            continuous_device_footprint_bytes(u64::MAX, u32::MAX, u64::MAX, u64::MAX, u64::MAX),
            u64::MAX
        );
    }

    #[test]
    fn vram_budget_rejects_oversized_gpu_plan() {
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
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
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_CUDA;
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.budget.max_comb_size = 2;
        config.budget.max_combinations_per_k = 1_000;
        // Default vram_budget_mb (6144) easily fits a small plan.
        assert!(prepare_continuous_execution(&config, 32, 8).is_ok());
    }
}
