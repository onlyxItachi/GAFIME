#[derive(Clone, Debug)]
pub struct ChunkRange {
    pub first_chunk_id: u32,
    pub chunk_count: u32,
    pub chunk_size: u64,
}

#[derive(Clone, Debug)]
pub struct ContinuousDescriptor {
    pub arity: u64,
    pub feature_start: u64,
    pub feature_stop: u64,
    pub universe_count: u128,
    pub planned_count: u128,
    pub offset: u64,
    pub chunk_range: ChunkRange,
    pub saturated: bool,
}

#[derive(Clone, Debug)]
pub struct DiscreteDescriptor {
    pub feature_start: u64,
    pub feature_stop: u64,
    pub threshold_count: u64,
    pub interval_count: u64,
    pub rectangle_pair_count: u64,
    pub template_count: u64,
    pub universe_count: u128,
    pub planned_count: u128,
    pub offset: u64,
    pub saturated: bool,
}

#[derive(Clone, Debug)]
pub struct TimeSeriesDescriptor {
    pub feature_start: u64,
    pub feature_stop: u64,
    pub lag_count: u64,
    pub window_count: u64,
    pub template_count: u64,
    pub universe_count: u128,
    pub planned_count: u128,
    pub offset: u64,
    pub saturated: bool,
}

#[derive(Clone, Debug)]
pub struct ScenarioPlan {
    pub n_samples: u64,
    pub n_features: u64,
    pub feature_candidate_count: u64,
    pub continuous: Vec<ContinuousDescriptor>,
    pub discrete: Option<DiscreteDescriptor>,
    pub time_series: Option<TimeSeriesDescriptor>,
    pub warnings: Vec<String>,
}

impl ScenarioPlan {
    pub fn empty(n_samples: u64, n_features: u64) -> Self {
        Self {
            n_samples,
            n_features,
            feature_candidate_count: n_features,
            continuous: Vec::new(),
            discrete: None,
            time_series: None,
            warnings: Vec::new(),
        }
    }
}

pub fn u128_text(value: u128) -> String {
    value.to_string()
}
