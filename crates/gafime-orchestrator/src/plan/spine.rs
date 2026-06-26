#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSpinePolicy {
    Unary,
    InteractionAware,
    DiversityReserve,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSpineConfig {
    pub policy: FeatureSpinePolicy,
    pub top_k: u32,
    pub diversity_reserve: u32,
}

impl Default for FeatureSpineConfig {
    fn default() -> Self {
        Self {
            policy: FeatureSpinePolicy::InteractionAware,
            top_k: 50,
            diversity_reserve: 8,
        }
    }
}
