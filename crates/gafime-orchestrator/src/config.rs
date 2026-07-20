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
    pub significance_top_n: u32,
    pub random_seed: u64,
    /// Absolute Python integer seed encoded as little-endian 32-bit words for
    /// legacy-compatible candidate planning. `random_seed` remains the bounded
    /// ABI seed used by significance backends.
    pub planning_seed_words: Vec<u32>,
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
            significance_top_n: 50,
            random_seed: raw.random_seed,
            planning_seed_words: Vec::new(),
            mi_bins: raw.mi_bins,
            mi_approximate: false,
            graph_requested: false,
        }
    }
}

impl EngineConfig {
    pub fn effective_feature_candidate_count(&self, feature_count: u32) -> u32 {
        match self.budget.max_feature_candidate {
            value if value >= 0 => feature_count.min(u32::try_from(value).unwrap_or(u32::MAX)),
            -1 if self.has_explicit_candidate_limits() => feature_count,
            -1 => feature_count.min(1_024),
            _ => feature_count,
        }
    }

    fn has_explicit_candidate_limits(&self) -> bool {
        let defaults = GafimeComputeBudget::default();
        self.budget.max_comb_size != defaults.max_comb_size
            || self.budget.max_combinations_per_k != defaults.max_combinations_per_k
            || self.budget.top_features_for_higher_k != defaults.top_features_for_higher_k
            || self.budget.max_time_series_candidates != defaults.max_time_series_candidates
            || self.budget.top_k_features_for_time_series != defaults.top_k_features_for_time_series
    }

    pub fn effective_planning_seed_words(&self) -> Vec<u32> {
        if !self.planning_seed_words.is_empty() {
            return self.planning_seed_words.clone();
        }
        if self.random_seed <= u32::MAX as u64 {
            vec![self.random_seed as u32]
        } else {
            vec![self.random_seed as u32, (self.random_seed >> 32) as u32]
        }
    }

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

    #[test]
    fn feature_candidate_limit_preserves_legacy_none_and_power_user_modes() {
        let mut config = EngineConfig::default();
        assert_eq!(config.effective_feature_candidate_count(2_000), 2_000);

        config.budget.max_feature_candidate = 7;
        assert_eq!(config.effective_feature_candidate_count(20), 7);
        config.budget.max_feature_candidate = 0;
        assert_eq!(config.effective_feature_candidate_count(20), 0);

        config.budget.max_feature_candidate = i64::from(u32::MAX) + 1;
        assert_eq!(config.effective_feature_candidate_count(20), 20);

        config.budget.max_feature_candidate = -1;
        assert_eq!(config.effective_feature_candidate_count(2_000), 1_024);
        config.budget.max_combinations_per_k = 2;
        assert_eq!(config.effective_feature_candidate_count(2_000), 2_000);
    }
}
