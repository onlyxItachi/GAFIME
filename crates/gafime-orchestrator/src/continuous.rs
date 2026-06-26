use gafime_types::{BackendKind, GafimeRankSpec, GAFIME_BACKEND_CPU, GAFIME_BACKEND_CUDA};

use crate::{
    config::EngineConfig,
    plan::{
        combos::{build_continuous_plan, ContinuousPlanRequest},
        CompiledPlan,
    },
    OrchestratorError, OrchestratorResult,
};

#[derive(Debug)]
pub struct PreparedContinuousExecution {
    plan: CompiledPlan,
    result_capacity: u64,
    result_max_arity: u32,
    result_metric_count: u32,
}

impl PreparedContinuousExecution {
    pub fn plan(&self) -> &CompiledPlan {
        &self.plan
    }

    pub fn into_plan(self) -> CompiledPlan {
        self.plan
    }

    pub fn result_capacity(&self) -> u64 {
        self.result_capacity
    }

    pub fn result_max_arity(&self) -> u32 {
        self.result_max_arity
    }

    pub fn result_metric_count(&self) -> u32 {
        self.result_metric_count
    }
}

pub fn continuous_backend_kind(config: &EngineConfig) -> OrchestratorResult<BackendKind> {
    match config.backend_kind {
        0 | GAFIME_BACKEND_CPU => Ok(GAFIME_BACKEND_CPU),
        GAFIME_BACKEND_CUDA => Ok(GAFIME_BACKEND_CUDA),
        _ => Err(OrchestratorError::Unsupported(
            "continuous v1 execution currently supports CPU and CUDA",
        )),
    }
}

pub fn prepare_continuous_execution(
    config: &EngineConfig,
    rows: u64,
    cols: u32,
) -> OrchestratorResult<PreparedContinuousExecution> {
    let backend_kind = continuous_backend_kind(config)?;
    let plan = build_continuous_plan(ContinuousPlanRequest {
        backend_kind,
        n_samples: rows,
        n_features: cols,
        max_arity: config.budget.max_comb_size,
        max_combinations_per_arity: config.budget.max_combinations_per_k,
        metric_ids: config.metric_ids.clone(),
        rank: GafimeRankSpec::default(),
    })?;
    Ok(PreparedContinuousExecution {
        result_capacity: plan.planned_row_count(),
        result_max_arity: plan.max_arity(),
        result_metric_count: plan.metric_count(),
        plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::{GAFIME_BACKEND_METAL, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2};

    #[test]
    fn default_config_prepares_cpu_continuous_execution() {
        let mut config = EngineConfig::default();
        config.metric_ids = vec![GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2];
        config.budget.max_comb_size = 3;
        config.budget.max_combinations_per_k = 1_000;

        let prepared = prepare_continuous_execution(&config, 32, 5).unwrap();

        assert_eq!(prepared.plan().protocol().backend_kind, GAFIME_BACKEND_CPU);
        assert_eq!(prepared.result_max_arity(), 3);
        assert_eq!(prepared.result_metric_count(), 2);
        assert_eq!(prepared.result_capacity(), 25);
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
        assert_eq!(prepared.result_capacity(), 10);
    }

    #[test]
    fn unsupported_backend_is_not_silently_fallbacked() {
        let mut config = EngineConfig::default();
        config.backend_kind = GAFIME_BACKEND_METAL;

        assert!(matches!(
            prepare_continuous_execution(&config, 32, 4),
            Err(OrchestratorError::Unsupported(_))
        ));
    }
}
