mod covariance;
mod covariance_common;
mod covariance_f32;
mod covariance_f64;
mod histogram;
mod isa;

pub use covariance::{pearson_corr, pearson_sums, pearson_sums_scalar, r2_score, PearsonSums};
#[cfg(test)]
pub(crate) use covariance_common::FP64_SIMD_REGROUPING_TOLERANCE;
pub(crate) use covariance_common::{finalize_correlation_f64, finalize_r2_f64};
pub use covariance_f32::pearson_corr_f32;
pub use covariance_f64::pearson_corr_f64;
pub use histogram::{fixed_bin_histogram2d, fixed_bin_indices, fixed_bin_indices_into};
pub use isa::{detect_isa, finite_dispatch_isa, IsaLevel};
