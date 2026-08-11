//! Time-series feature expansion for the `time_series` candidate family.
//!
//! Given a column-major feature matrix (column `c` is
//! `columns[c*rows .. (c+1)*rows]`), this generates derived columns — lag,
//! delta, velocity, acceleration, rolling mean/std/sum — which the continuous
//! engine then scores against the target. Positions without enough history are
//! filled with `f32::NAN` so the finite-pair scoring skips them (no leakage, no
//! bias).

use gafime_orchestrator::{OrchestratorError, OrchestratorResult};
use gafime_types::PrecisionProfile;

use crate::precision::{CpuPrecisionSlice, CpuPrecisionValues};

/// One generated time-series operation on a base feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeSeriesOp {
    /// x[t-lag]
    Lag(u32),
    /// x[t] - x[t-lag]
    Delta(u32),
    /// (x[t] - x[t-lag]) / lag
    Velocity(u32),
    /// (x[t] - 2*x[t-lag] + x[t-2*lag]) / lag^2
    Acceleration(u32),
    /// mean(x[t-window+1 ..= t])
    RollingMean(u32),
    /// population std(x[t-window+1 ..= t])
    RollingStd(u32),
    /// sum(x[t-window+1 ..= t])
    RollingSum(u32),
}

/// Descriptor of a generated time-series feature column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeSeriesFeature {
    pub base_feature: u32,
    pub op: TimeSeriesOp,
}

fn finalize_rolling_variance_f32(variance: f32) -> f32 {
    if variance.is_finite() {
        variance.max(0.0)
    } else {
        f32::NAN
    }
}

fn finalize_rolling_variance_f64(variance: f64) -> f64 {
    if variance.is_finite() {
        variance.max(0.0)
    } else {
        f64::NAN
    }
}

/// Generate time-series feature columns (column-major, `rows` each) from the
/// base `columns`. Returns the flat column-major data and a descriptor per
/// generated column (in emission order).
pub fn time_series_columns(
    columns: &[f32],
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> (Vec<f32>, Vec<TimeSeriesFeature>) {
    let mut out: Vec<f32> = Vec::new();
    let mut descriptors: Vec<TimeSeriesFeature> = Vec::new();
    if rows == 0 || cols == 0 {
        return (out, descriptors);
    }

    for base in 0..cols {
        let col = &columns[base * rows..(base + 1) * rows];

        for &lag in lags {
            let k = lag as usize;
            if k == 0 || k >= rows {
                continue; // no usable history
            }
            let mut lag_feature = vec![f32::NAN; rows];
            lag_feature[k..rows].copy_from_slice(&col[..rows - k]);
            out.extend_from_slice(&lag_feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::Lag(lag),
            });

            if velocity {
                let mut delta_feature = vec![f32::NAN; rows];
                let mut velocity_feature = vec![f32::NAN; rows];
                let scale = k as f32;
                for t in k..rows {
                    let delta = col[t] - col[t - k];
                    delta_feature[t] = delta;
                    velocity_feature[t] = delta / scale;
                }
                out.extend_from_slice(&delta_feature);
                descriptors.push(TimeSeriesFeature {
                    base_feature: base as u32,
                    op: TimeSeriesOp::Delta(lag),
                });
                out.extend_from_slice(&velocity_feature);
                descriptors.push(TimeSeriesFeature {
                    base_feature: base as u32,
                    op: TimeSeriesOp::Velocity(lag),
                });

                let two_k = k.saturating_mul(2);
                if two_k < rows {
                    let mut acceleration_feature = vec![f32::NAN; rows];
                    let scale2 = (k * k) as f32;
                    for t in two_k..rows {
                        acceleration_feature[t] =
                            (col[t] - 2.0 * col[t - k] + col[t - two_k]) / scale2;
                    }
                    out.extend_from_slice(&acceleration_feature);
                    descriptors.push(TimeSeriesFeature {
                        base_feature: base as u32,
                        op: TimeSeriesOp::Acceleration(lag),
                    });
                }
            }
        }

        for &window in windows {
            let w = window as usize;
            if w < 2 || w > rows {
                continue;
            }
            let mut mean_feature = vec![f32::NAN; rows];
            let mut std_feature = vec![f32::NAN; rows];
            let mut sum_feature = vec![f32::NAN; rows];
            // Running sum in f64 (consistent with the centered-covariance fp32
            // policy: fp32 storage, higher-precision reduction).
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut invalid = 0usize;
            for t in 0..rows {
                let value = col[t];
                if value.is_finite() {
                    let v = value as f64;
                    sum += v;
                    sum2 += v * v;
                } else {
                    invalid += 1;
                }
                if t >= w {
                    let old = col[t - w];
                    if old.is_finite() {
                        let v = old as f64;
                        sum -= v;
                        sum2 -= v * v;
                    } else {
                        invalid -= 1;
                    }
                }
                if t + 1 >= w && invalid == 0 {
                    let mean = sum / w as f64;
                    let variance = finalize_rolling_variance_f64(sum2 / w as f64 - mean * mean);
                    mean_feature[t] = mean as f32;
                    std_feature[t] = variance.sqrt() as f32;
                    sum_feature[t] = sum as f32;
                }
            }
            out.extend_from_slice(&mean_feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingMean(window),
            });
            out.extend_from_slice(&std_feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingStd(window),
            });
            out.extend_from_slice(&sum_feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingSum(window),
            });
        }
    }

    (out, descriptors)
}

/// Generate time-series columns in the selected precision profile.
///
/// Scheduling descriptors, lags, windows, and emitted ordering stay integer
/// and identical between lanes.  Only numeric generated values vary: fp32
/// executes every operation in f32, mixed owns f32 generated storage while
/// widening rolling reductions, and fp64 never visits an f32 buffer.
pub fn time_series_columns_precision(
    profile: PrecisionProfile,
    columns: CpuPrecisionSlice<'_>,
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> OrchestratorResult<(CpuPrecisionValues, Vec<TimeSeriesFeature>)> {
    match (profile, columns) {
        (PrecisionProfile::Fp32, CpuPrecisionSlice::F32(columns)) => {
            let (values, descriptors) =
                time_series_columns_f32(columns, rows, cols, lags, windows, velocity);
            Ok((CpuPrecisionValues::F32(values), descriptors))
        }
        (PrecisionProfile::Mixed, CpuPrecisionSlice::F32(columns)) => {
            let (values, descriptors) =
                time_series_columns_mixed(columns, rows, cols, lags, windows, velocity);
            Ok((CpuPrecisionValues::F32(values), descriptors))
        }
        (PrecisionProfile::Fp64, CpuPrecisionSlice::F64(columns)) => {
            let (values, descriptors) =
                time_series_columns_f64(columns, rows, cols, lags, windows, velocity);
            Ok((CpuPrecisionValues::F64(values), descriptors))
        }
        _ => Err(OrchestratorError::InvalidPlan(
            "time-series storage dtype does not match the requested precision profile",
        )),
    }
}

fn time_series_columns_f32(
    columns: &[f32],
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> (Vec<f32>, Vec<TimeSeriesFeature>) {
    let mut out = Vec::new();
    let mut descriptors = Vec::new();
    if rows == 0 || cols == 0 {
        return (out, descriptors);
    }
    for base in 0..cols {
        let col = &columns[base * rows..(base + 1) * rows];
        emit_lag_columns_f32(
            col,
            rows,
            base as u32,
            lags,
            velocity,
            &mut out,
            &mut descriptors,
        );
        for &window in windows {
            let w = window as usize;
            if w < 2 || w > rows {
                continue;
            }
            let mut mean = vec![f32::NAN; rows];
            let mut std = vec![f32::NAN; rows];
            let mut sum_out = vec![f32::NAN; rows];
            let mut sum = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut invalid = 0usize;
            for t in 0..rows {
                let value = col[t];
                if value.is_finite() {
                    sum += value;
                    sum2 += value * value;
                } else {
                    invalid += 1;
                }
                if t >= w {
                    let old = col[t - w];
                    if old.is_finite() {
                        sum -= old;
                        sum2 -= old * old;
                    } else {
                        invalid -= 1;
                    }
                }
                if t + 1 >= w && invalid == 0 {
                    let current_mean = sum / w as f32;
                    let variance = finalize_rolling_variance_f32(
                        sum2 / w as f32 - current_mean * current_mean,
                    );
                    mean[t] = current_mean;
                    std[t] = variance.sqrt();
                    sum_out[t] = sum;
                }
            }
            emit_rolling_columns(
                base as u32,
                window,
                mean,
                std,
                sum_out,
                &mut out,
                &mut descriptors,
            );
        }
    }
    (out, descriptors)
}

fn time_series_columns_mixed(
    columns: &[f32],
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> (Vec<f32>, Vec<TimeSeriesFeature>) {
    let mut out = Vec::new();
    let mut descriptors = Vec::new();
    if rows == 0 || cols == 0 {
        return (out, descriptors);
    }
    for base in 0..cols {
        let col = &columns[base * rows..(base + 1) * rows];
        // These are pointwise f32 transforms and remain byte-for-byte f32.
        emit_lag_columns_f32(
            col,
            rows,
            base as u32,
            lags,
            velocity,
            &mut out,
            &mut descriptors,
        );
        for &window in windows {
            let w = window as usize;
            if w < 2 || w > rows {
                continue;
            }
            let mut mean = vec![f32::NAN; rows];
            let mut std = vec![f32::NAN; rows];
            let mut sum_out = vec![f32::NAN; rows];
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut invalid = 0usize;
            for t in 0..rows {
                let value = col[t];
                if value.is_finite() {
                    let value = value as f64;
                    sum += value;
                    sum2 += value * value;
                } else {
                    invalid += 1;
                }
                if t >= w {
                    let old = col[t - w];
                    if old.is_finite() {
                        let old = old as f64;
                        sum -= old;
                        sum2 -= old * old;
                    } else {
                        invalid -= 1;
                    }
                }
                if t + 1 >= w && invalid == 0 {
                    let current_mean = sum / w as f64;
                    let variance = finalize_rolling_variance_f64(
                        sum2 / w as f64 - current_mean * current_mean,
                    );
                    // Generated storage is a materialized f32 feature in mixed.
                    mean[t] = current_mean as f32;
                    std[t] = variance.sqrt() as f32;
                    sum_out[t] = sum as f32;
                }
            }
            emit_rolling_columns(
                base as u32,
                window,
                mean,
                std,
                sum_out,
                &mut out,
                &mut descriptors,
            );
        }
    }
    (out, descriptors)
}

fn time_series_columns_f64(
    columns: &[f64],
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> (Vec<f64>, Vec<TimeSeriesFeature>) {
    let mut out = Vec::new();
    let mut descriptors = Vec::new();
    if rows == 0 || cols == 0 {
        return (out, descriptors);
    }
    for base in 0..cols {
        let col = &columns[base * rows..(base + 1) * rows];
        emit_lag_columns_f64(
            col,
            rows,
            base as u32,
            lags,
            velocity,
            &mut out,
            &mut descriptors,
        );
        for &window in windows {
            let w = window as usize;
            if w < 2 || w > rows {
                continue;
            }
            let mut mean = vec![f64::NAN; rows];
            let mut std = vec![f64::NAN; rows];
            let mut sum_out = vec![f64::NAN; rows];
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut invalid = 0usize;
            for t in 0..rows {
                let value = col[t];
                if value.is_finite() {
                    sum += value;
                    sum2 += value * value;
                } else {
                    invalid += 1;
                }
                if t >= w {
                    let old = col[t - w];
                    if old.is_finite() {
                        sum -= old;
                        sum2 -= old * old;
                    } else {
                        invalid -= 1;
                    }
                }
                if t + 1 >= w && invalid == 0 {
                    let current_mean = sum / w as f64;
                    let variance = finalize_rolling_variance_f64(
                        sum2 / w as f64 - current_mean * current_mean,
                    );
                    mean[t] = current_mean;
                    std[t] = variance.sqrt();
                    sum_out[t] = sum;
                }
            }
            out.extend_from_slice(&mean);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingMean(window),
            });
            out.extend_from_slice(&std);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingStd(window),
            });
            out.extend_from_slice(&sum_out);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingSum(window),
            });
        }
    }
    (out, descriptors)
}

fn emit_lag_columns_f32(
    col: &[f32],
    rows: usize,
    base: u32,
    lags: &[u32],
    velocity: bool,
    out: &mut Vec<f32>,
    descriptors: &mut Vec<TimeSeriesFeature>,
) {
    for &lag in lags {
        let k = lag as usize;
        if k == 0 || k >= rows {
            continue;
        }
        let mut lag_feature = vec![f32::NAN; rows];
        lag_feature[k..].copy_from_slice(&col[..rows - k]);
        out.extend_from_slice(&lag_feature);
        descriptors.push(TimeSeriesFeature {
            base_feature: base,
            op: TimeSeriesOp::Lag(lag),
        });
        if velocity {
            let mut delta = vec![f32::NAN; rows];
            let mut velocity_values = vec![f32::NAN; rows];
            let scale = k as f32;
            for t in k..rows {
                delta[t] = col[t] - col[t - k];
                velocity_values[t] = delta[t] / scale;
            }
            out.extend_from_slice(&delta);
            descriptors.push(TimeSeriesFeature {
                base_feature: base,
                op: TimeSeriesOp::Delta(lag),
            });
            out.extend_from_slice(&velocity_values);
            descriptors.push(TimeSeriesFeature {
                base_feature: base,
                op: TimeSeriesOp::Velocity(lag),
            });
            let two_k = k.saturating_mul(2);
            if two_k < rows {
                let mut acceleration = vec![f32::NAN; rows];
                let scale2 = (k * k) as f32;
                for t in two_k..rows {
                    acceleration[t] = (col[t] - 2.0 * col[t - k] + col[t - two_k]) / scale2;
                }
                out.extend_from_slice(&acceleration);
                descriptors.push(TimeSeriesFeature {
                    base_feature: base,
                    op: TimeSeriesOp::Acceleration(lag),
                });
            }
        }
    }
}

fn emit_lag_columns_f64(
    col: &[f64],
    rows: usize,
    base: u32,
    lags: &[u32],
    velocity: bool,
    out: &mut Vec<f64>,
    descriptors: &mut Vec<TimeSeriesFeature>,
) {
    for &lag in lags {
        let k = lag as usize;
        if k == 0 || k >= rows {
            continue;
        }
        let mut lag_feature = vec![f64::NAN; rows];
        lag_feature[k..].copy_from_slice(&col[..rows - k]);
        out.extend_from_slice(&lag_feature);
        descriptors.push(TimeSeriesFeature {
            base_feature: base,
            op: TimeSeriesOp::Lag(lag),
        });
        if velocity {
            let mut delta = vec![f64::NAN; rows];
            let mut velocity_values = vec![f64::NAN; rows];
            let scale = k as f64;
            for t in k..rows {
                delta[t] = col[t] - col[t - k];
                velocity_values[t] = delta[t] / scale;
            }
            out.extend_from_slice(&delta);
            descriptors.push(TimeSeriesFeature {
                base_feature: base,
                op: TimeSeriesOp::Delta(lag),
            });
            out.extend_from_slice(&velocity_values);
            descriptors.push(TimeSeriesFeature {
                base_feature: base,
                op: TimeSeriesOp::Velocity(lag),
            });
            let two_k = k.saturating_mul(2);
            if two_k < rows {
                let mut acceleration = vec![f64::NAN; rows];
                let scale2 = (k * k) as f64;
                for t in two_k..rows {
                    acceleration[t] = (col[t] - 2.0 * col[t - k] + col[t - two_k]) / scale2;
                }
                out.extend_from_slice(&acceleration);
                descriptors.push(TimeSeriesFeature {
                    base_feature: base,
                    op: TimeSeriesOp::Acceleration(lag),
                });
            }
        }
    }
}

fn emit_rolling_columns(
    base: u32,
    window: u32,
    mean: Vec<f32>,
    std: Vec<f32>,
    sum: Vec<f32>,
    out: &mut Vec<f32>,
    descriptors: &mut Vec<TimeSeriesFeature>,
) {
    out.extend_from_slice(&mean);
    descriptors.push(TimeSeriesFeature {
        base_feature: base,
        op: TimeSeriesOp::RollingMean(window),
    });
    out.extend_from_slice(&std);
    descriptors.push(TimeSeriesFeature {
        base_feature: base,
        op: TimeSeriesOp::RollingStd(window),
    });
    out.extend_from_slice(&sum);
    descriptors.push(TimeSeriesFeature {
        base_feature: base,
        op: TimeSeriesOp::RollingSum(window),
    });
}

/// Expand a row-major feature matrix in place-of-mining: the time-series columns
/// are appended after the `cols` base features. Returns (expanded row-major,
/// expanded column count, descriptors in appended order). The continuous engine
/// then mines the expanded matrix on whichever backend (CPU/GPU) — TS candidates
/// are just additional continuous features.
pub fn expand_row_major(
    features: &[f32],
    rows: usize,
    cols: usize,
    lags: &[u32],
    windows: &[u32],
    velocity: bool,
) -> (Vec<f32>, usize, Vec<TimeSeriesFeature>) {
    if rows == 0 || cols == 0 {
        return (features.to_vec(), cols, Vec::new());
    }
    // row-major -> column-major for the per-column time-series ops
    let mut colmajor = vec![0.0f32; rows * cols];
    for t in 0..rows {
        let base = t * cols;
        for c in 0..cols {
            colmajor[c * rows + t] = features[base + c];
        }
    }
    let (ts_cols, descriptors) =
        time_series_columns(&colmajor, rows, cols, lags, windows, velocity);
    let n_ts = descriptors.len();
    let ecols = cols + n_ts;
    let mut expanded = vec![0.0f32; rows * ecols];
    for t in 0..rows {
        let src = t * cols;
        let dst = t * ecols;
        expanded[dst..dst + cols].copy_from_slice(&features[src..src + cols]);
        for j in 0..n_ts {
            expanded[dst + cols + j] = ts_cols[j * rows + t];
        }
    }
    (expanded, ecols, descriptors)
}

/// Human-readable label for a generated feature, e.g. `sales_lag4`.
pub fn feature_label(base_name: &str, op: TimeSeriesOp) -> String {
    match op {
        TimeSeriesOp::Lag(k) => format!("{base_name}_lag{k}"),
        TimeSeriesOp::Delta(k) => format!("{base_name}_delta{k}"),
        TimeSeriesOp::Velocity(k) => format!("{base_name}_velocity{k}"),
        TimeSeriesOp::Acceleration(k) => format!("{base_name}_acceleration{k}"),
        TimeSeriesOp::RollingMean(w) => format!("{base_name}_rollmean{w}"),
        TimeSeriesOp::RollingStd(w) => format!("{base_name}_rollstd{w}"),
        TimeSeriesOp::RollingSum(w) => format!("{base_name}_rollsum{w}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_time_series_keeps_profile_storage_and_reduction_contracts() {
        let columns = [16_777_216.0f32, 1.0, -16_777_216.0, 2.0];
        let (fp32, descriptors) = time_series_columns_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&columns),
            4,
            1,
            &[],
            &[3],
            false,
        )
        .unwrap();
        let (mixed, mixed_descriptors) = time_series_columns_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&columns),
            4,
            1,
            &[],
            &[3],
            false,
        )
        .unwrap();
        assert_eq!(descriptors, mixed_descriptors);
        let CpuPrecisionValues::F32(fp32) = fp32 else {
            panic!("fp32 stays f32")
        };
        let CpuPrecisionValues::F32(mixed) = mixed else {
            panic!("mixed storage stays f32")
        };
        // The first rolling-mean output is at index two.  fp32's running sum
        // loses the unit, mixed widens only the reduction then materializes f32.
        assert_eq!(fp32[2], 0.0);
        assert_eq!(mixed[2], 1.0 / 3.0);
    }

    #[test]
    fn fp64_time_series_preserves_adjacent_binary64_values() {
        let base = 1.0f64;
        let next = f64::from_bits(base.to_bits() + 1);
        let columns = [base, next, f64::from_bits(base.to_bits() + 2)];
        let (result, descriptors) = time_series_columns_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&columns),
            3,
            1,
            &[1],
            &[],
            false,
        )
        .unwrap();
        assert_eq!(descriptors[0].op, TimeSeriesOp::Lag(1));
        let CpuPrecisionValues::F64(values) = result else {
            panic!("fp64 stays f64")
        };
        assert_eq!(values[1].to_bits(), base.to_bits());
        assert_eq!(values[2].to_bits(), next.to_bits());
    }

    #[test]
    fn precision_time_series_rejects_storage_mismatch_before_transforming() {
        assert!(time_series_columns_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F32(&[1.0, 2.0]),
            2,
            1,
            &[1],
            &[],
            false,
        )
        .is_err());
    }

    #[test]
    fn rolling_variance_preserves_nonfinite_failure_and_mixed_remains_finite() {
        let fp32_columns = [f32::MAX, f32::MAX / 2.0];
        let (fp32, _) = time_series_columns_precision(
            PrecisionProfile::Fp32,
            CpuPrecisionSlice::F32(&fp32_columns),
            2,
            1,
            &[],
            &[2],
            false,
        )
        .unwrap();
        let CpuPrecisionValues::F32(fp32) = fp32 else {
            panic!("fp32 rolling features must stay f32")
        };
        assert!(fp32[3].is_nan());

        let (mixed, _) = time_series_columns_precision(
            PrecisionProfile::Mixed,
            CpuPrecisionSlice::F32(&fp32_columns),
            2,
            1,
            &[],
            &[2],
            false,
        )
        .unwrap();
        let CpuPrecisionValues::F32(mixed) = mixed else {
            panic!("mixed rolling storage must stay f32")
        };
        assert!(mixed[3].is_finite());
        assert!(mixed[3] > 0.0);

        let fp64_columns = [f64::MAX, f64::MAX / 2.0];
        let (fp64, _) = time_series_columns_precision(
            PrecisionProfile::Fp64,
            CpuPrecisionSlice::F64(&fp64_columns),
            2,
            1,
            &[],
            &[2],
            false,
        )
        .unwrap();
        let CpuPrecisionValues::F64(fp64) = fp64 else {
            panic!("fp64 rolling features must stay f64")
        };
        assert!(fp64[3].is_nan());
    }

    #[test]
    fn rolling_variance_clamps_only_finite_negative_roundoff() {
        assert_eq!(finalize_rolling_variance_f32(-f32::EPSILON), 0.0);
        assert_eq!(finalize_rolling_variance_f64(-f64::EPSILON), 0.0);
        assert!(finalize_rolling_variance_f32(f32::NAN).is_nan());
        assert!(finalize_rolling_variance_f32(f32::INFINITY).is_nan());
        assert!(finalize_rolling_variance_f64(f64::NAN).is_nan());
        assert!(finalize_rolling_variance_f64(f64::INFINITY).is_nan());
    }

    #[test]
    fn expand_appends_ts_columns_after_base() {
        // 1 base feature, 4 rows: [1,2,3,4]; lag1 + delta/velocity/accel -> 4 derived cols
        let f = vec![1.0f32, 2.0, 3.0, 4.0];
        let (exp, ecols, desc) = expand_row_major(&f, 4, 1, &[1], &[], true);
        assert_eq!(ecols, 5);
        assert_eq!(desc.len(), 4);
        // row t=2: base=3, lag1=2, delta1=1, velocity1=1, acceleration1=0
        assert_eq!(exp[2 * 5], 3.0);
        assert_eq!(exp[2 * 5 + 1], 2.0);
        assert_eq!(exp[2 * 5 + 2], 1.0);
        assert_eq!(exp[2 * 5 + 3], 1.0);
        assert_eq!(exp[2 * 5 + 4], 0.0);
        assert_eq!(feature_label("sales", desc[0].op), "sales_lag1");
    }

    #[test]
    fn lag_shifts_and_nans_the_boundary() {
        let cols = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (out, desc) = time_series_columns(&cols, 5, 1, &[2], &[], false);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(&out[2..5], &[1.0, 2.0, 3.0]); // x[t-2]
        assert_eq!(
            desc,
            vec![TimeSeriesFeature {
                base_feature: 0,
                op: TimeSeriesOp::Lag(2)
            }]
        );
    }

    #[test]
    fn rolling_mean_averages_the_window() {
        let cols = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (out, _) = time_series_columns(&cols, 5, 1, &[], &[3], false);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(&out[2..5], &[2.0, 3.0, 4.0]); // mean of trailing 3
    }

    #[test]
    fn lag_derivatives_use_requested_lag_and_nan_boundary() {
        let cols = vec![1.0f32, 3.0, 6.0, 10.0, 15.0];
        let (out, desc) = time_series_columns(&cols, 5, 1, &[2], &[], true);
        assert_eq!(
            desc.iter().map(|item| item.op).collect::<Vec<_>>(),
            vec![
                TimeSeriesOp::Lag(2),
                TimeSeriesOp::Delta(2),
                TimeSeriesOp::Velocity(2),
                TimeSeriesOp::Acceleration(2),
            ]
        );
        let lag = &out[0..5];
        let delta = &out[5..10];
        let velocity = &out[10..15];
        let acceleration = &out[15..20];
        assert!(lag[0].is_nan() && lag[1].is_nan());
        assert_eq!(&lag[2..5], &[1.0, 3.0, 6.0]);
        assert!(delta[0].is_nan() && delta[1].is_nan());
        assert_eq!(&delta[2..5], &[5.0, 7.0, 9.0]);
        assert_eq!(&velocity[2..5], &[2.5, 3.5, 4.5]);
        assert!(acceleration[0].is_nan() && acceleration[3].is_nan());
        assert_eq!(acceleration[4], 1.0);
    }

    #[test]
    fn rolling_std_and_sum_follow_the_same_window() {
        let cols = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (out, desc) = time_series_columns(&cols, 5, 1, &[], &[3], false);
        assert_eq!(
            desc.iter().map(|item| item.op).collect::<Vec<_>>(),
            vec![
                TimeSeriesOp::RollingMean(3),
                TimeSeriesOp::RollingStd(3),
                TimeSeriesOp::RollingSum(3),
            ]
        );
        let std = &out[5..10];
        let sum = &out[10..15];
        assert!(std[0].is_nan() && std[1].is_nan());
        assert!((std[2] - (2.0f32 / 3.0).sqrt()).abs() < 1e-6);
        assert_eq!(&sum[2..5], &[6.0, 9.0, 12.0]);
    }

    #[test]
    fn multi_feature_emission_order_and_count() {
        // 2 cols x 3 rows; lag1 + three lag derivatives -> 4 generated cols per base = 8
        let cols = vec![1.0f32, 2.0, 3.0, /*c0*/ 10.0, 20.0, 30.0 /*c1*/];
        let (out, desc) = time_series_columns(&cols, 3, 2, &[1], &[], true);
        assert_eq!(desc.len(), 8);
        assert_eq!(out.len(), 8 * 3);
        assert_eq!(desc[0].base_feature, 0);
        assert_eq!(desc[0].op, TimeSeriesOp::Lag(1));
        assert_eq!(desc[4].base_feature, 1);
    }
}
