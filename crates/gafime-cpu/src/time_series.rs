//! Time-series feature expansion for the `time_series` candidate family.
//!
//! Given a column-major feature matrix (column `c` is
//! `columns[c*rows .. (c+1)*rows]`), this generates derived columns — lag,
//! delta, velocity, acceleration, rolling mean/std/sum — which the continuous
//! engine then scores against the target. Positions without enough history are
//! filled with `f32::NAN` so the finite-pair scoring skips them (no leakage, no
//! bias).

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
            for t in k..rows {
                lag_feature[t] = col[t - k];
            }
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
                    let variance = (sum2 / w as f64 - mean * mean).max(0.0);
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
        for c in 0..cols {
            expanded[dst + c] = features[src + c];
        }
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
