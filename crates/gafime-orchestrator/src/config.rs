use gafime_types::{
    BackendKind, GafimeComputeBudget, GafimeEngineConfig, GAFIME_BACKEND_CPU,
    GAFIME_METRIC_MUTUAL_INFO, GAFIME_METRIC_PEARSON, GAFIME_METRIC_R2, GAFIME_METRIC_SPEARMAN,
};

#[derive(Clone, Debug, PartialEq)]
pub struct EngineConfig {
    pub backend_kind: BackendKind,
    pub device_id: u32,
    pub metric_ids: Vec<u32>,
    pub budget: GafimeComputeBudget,
    pub num_repeats: u32,
    pub permutation_tests: u32,
    pub random_seed: u64,
    /// Adaptive maximum for mutual-information histogram planning.
    pub mi_bins: u32,
    /// Opt-in: use the fixed-bin MI approximation backend (matches the GPU) on the
    /// CPU instead of the default adaptive-quantile MI.
    pub mi_approximate: bool,
    /// Request backend-native graph capture/replay for supported GPU backends.
    pub graph_requested: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let raw = GafimeEngineConfig::default();
        Self {
            backend_kind: GAFIME_BACKEND_CPU,
            device_id: raw.device_id,
            metric_ids: default_metric_ids(),
            budget: raw.budget,
            num_repeats: raw.num_repeats,
            permutation_tests: raw.permutation_tests,
            random_seed: raw.random_seed,
            mi_bins: raw.mi_bins,
            mi_approximate: false,
            graph_requested: false,
        }
    }
}

impl EngineConfig {
    pub fn bind_raw_views(&self, raw: &mut GafimeEngineConfig) {
        raw.backend_kind = self.backend_kind;
        raw.device_id = self.device_id;
        raw.metric_ids = gafime_types::GafimeSliceU32 {
            ptr: self.metric_ids.as_ptr(),
            len: self.metric_ids.len() as u64,
        };
        raw.budget = self.budget;
        raw.num_repeats = self.num_repeats;
        raw.permutation_tests = self.permutation_tests;
        raw.random_seed = self.random_seed;
        raw.mi_bins = self.mi_bins;
    }
}

pub fn default_metric_ids() -> Vec<u32> {
    vec![
        GAFIME_METRIC_PEARSON,
        GAFIME_METRIC_SPEARMAN,
        GAFIME_METRIC_MUTUAL_INFO,
        GAFIME_METRIC_R2,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use gafime_types::GAFIME_ABI_VERSION;

    #[test]
    fn owned_config_binds_to_raw_abi_view() {
        let config = EngineConfig::default();
        let mut raw = GafimeEngineConfig::default();
        config.bind_raw_views(&mut raw);

        assert_eq!(raw.abi_version, GAFIME_ABI_VERSION);
        assert_eq!(raw.backend_kind, GAFIME_BACKEND_CPU);
        assert_eq!(raw.metric_ids.len, 4);
        assert!(!raw.metric_ids.ptr.is_null());
    }
}
