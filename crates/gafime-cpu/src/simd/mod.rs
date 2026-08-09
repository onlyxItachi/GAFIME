mod covariance;
mod covariance_f32;
mod histogram;
mod isa;

pub use covariance::{pearson_corr, pearson_sums, pearson_sums_scalar, r2_score, PearsonSums};
pub use covariance_f32::pearson_corr_f32;
pub use histogram::{fixed_bin_histogram2d, fixed_bin_indices, fixed_bin_indices_into};
pub use isa::{detect_isa, finite_dispatch_isa, IsaLevel};
