//! Time-series feature expansion for the `time_series` candidate family.
//!
//! Given a column-major feature matrix (column `c` is
//! `columns[c*rows .. (c+1)*rows]`), this generates derived columns — lag,
//! rolling-mean, and first-difference (velocity) — which the continuous engine
//! then scores against the target. Positions without enough history are filled
//! with `f32::NAN` so the finite-pair scoring skips them (no leakage, no bias).

/// One generated time-series operation on a base feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeSeriesOp {
    /// x[t-lag]
    Lag(u32),
    /// mean(x[t-window+1 ..= t])
    RollingMean(u32),
    /// x[t] - x[t-1]
    Velocity,
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
            let mut feature = vec![f32::NAN; rows];
            for t in k..rows {
                feature[t] = col[t - k];
            }
            out.extend_from_slice(&feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::Lag(lag),
            });
        }

        for &window in windows {
            let w = window as usize;
            if w < 2 || w > rows {
                continue;
            }
            let mut feature = vec![f32::NAN; rows];
            // Running sum in f64 (consistent with the centered-covariance fp32
            // policy: fp32 storage, higher-precision reduction).
            let mut sum = 0.0f64;
            for t in 0..rows {
                sum += col[t] as f64;
                if t >= w {
                    sum -= col[t - w] as f64;
                }
                if t + 1 >= w {
                    feature[t] = (sum / w as f64) as f32;
                }
            }
            out.extend_from_slice(&feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::RollingMean(window),
            });
        }

        if velocity {
            let mut feature = vec![f32::NAN; rows];
            for t in 1..rows {
                feature[t] = col[t] - col[t - 1];
            }
            out.extend_from_slice(&feature);
            descriptors.push(TimeSeriesFeature {
                base_feature: base as u32,
                op: TimeSeriesOp::Velocity,
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
    let (ts_cols, descriptors) = time_series_columns(&colmajor, rows, cols, lags, windows, velocity);
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
        TimeSeriesOp::RollingMean(w) => format!("{base_name}_rollmean{w}"),
        TimeSeriesOp::Velocity => format!("{base_name}_velocity"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_appends_ts_columns_after_base() {
        // 1 base feature, 4 rows: [1,2,3,4]; lag1 + velocity -> 2 derived cols
        let f = vec![1.0f32, 2.0, 3.0, 4.0];
        let (exp, ecols, desc) = expand_row_major(&f, 4, 1, &[1], &[], true);
        assert_eq!(ecols, 3);
        assert_eq!(desc.len(), 2);
        // row t=1: base=2, lag1=x[0]=1, velocity=x[1]-x[0]=1
        assert_eq!(exp[1 * 3], 2.0);
        assert_eq!(exp[1 * 3 + 1], 1.0);
        assert_eq!(exp[1 * 3 + 2], 1.0);
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
    fn velocity_is_first_difference() {
        let cols = vec![1.0f32, 3.0, 6.0, 10.0];
        let (out, _) = time_series_columns(&cols, 4, 1, &[], &[], true);
        assert!(out[0].is_nan());
        assert_eq!(&out[1..4], &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn multi_feature_emission_order_and_count() {
        // 2 cols x 3 rows; lag 1 + velocity -> 2 generated cols per base = 4
        let cols = vec![1.0f32, 2.0, 3.0, /*c0*/ 10.0, 20.0, 30.0 /*c1*/];
        let (out, desc) = time_series_columns(&cols, 3, 2, &[1], &[], true);
        assert_eq!(desc.len(), 4);
        assert_eq!(out.len(), 4 * 3);
        assert_eq!(desc[0].base_feature, 0);
        assert_eq!(desc[0].op, TimeSeriesOp::Lag(1));
        assert_eq!(desc[2].base_feature, 1);
    }
}
