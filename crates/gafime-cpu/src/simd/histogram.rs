#[inline]
fn fixed_bin_from_scaled(scaled: f32, max_bin: u32) -> u32 {
    if scaled.is_nan() || scaled <= 0.0 {
        0
    } else if !scaled.is_finite() || scaled >= max_bin as f32 {
        max_bin
    } else {
        scaled as u32
    }
}

#[inline]
fn fixed_bin_index(value: f32, min: f32, inv: f32, max_bin: u32) -> u32 {
    fixed_bin_from_scaled((value - min) * inv, max_bin)
}

/// Fixed equal-width bin indices for `values`: `bin = clamp(trunc((v - min) *
/// inv), 0, bins-1)`, matching the GPU MI kernels' `f32` mapping. Overflow is
/// explicit: positive infinity maps to the final bin and NaN/negative infinity
/// map to zero. SIMD vectorizes the arithmetic and applies the same lane helper
/// as the scalar path before histogram scatter.
pub fn fixed_bin_indices(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2 is available;
            // the implementation reads and writes only complete in-bounds lanes.
            return unsafe { fixed_bin_indices_avx2(values, min, inv, bins) };
        }
    }
    fixed_bin_indices_scalar(values, min, inv, bins)
}

fn fixed_bin_indices_scalar(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    let max_bin = bins.saturating_sub(1);
    values
        .iter()
        .map(|&value| fixed_bin_index(value, min, inv, max_bin))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fixed_bin_indices_avx2(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    use std::arch::x86_64::*;

    let n = values.len();
    let mut out = vec![0u32; n];
    let max_bin = bins.saturating_sub(1);
    let min_v = _mm256_set1_ps(min);
    let inv_v = _mm256_set1_ps(inv);
    let mut scaled_lanes = [0.0f32; 8];

    let chunks = n / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let values_v = _mm256_loadu_ps(values.as_ptr().add(offset));
        let scaled = _mm256_mul_ps(_mm256_sub_ps(values_v, min_v), inv_v);
        _mm256_storeu_ps(scaled_lanes.as_mut_ptr(), scaled);
        for lane in 0..8 {
            out[offset + lane] = fixed_bin_from_scaled(scaled_lanes[lane], max_bin);
        }
    }
    for index in (chunks * 8)..n {
        out[index] = fixed_bin_index(values[index], min, inv, max_bin);
    }
    out
}

/// Build fixed-bin marginal and joint histograms for the GPU-compatible MI
/// approximation. AVX2 vectorizes the bin arithmetic and feeds lane-local scalar
/// histogram increments; the scatter itself is data-dependent and remains scalar.
#[allow(
    clippy::too_many_arguments,
    reason = "the hot path exposes two bin transforms and three independent histogram buffers explicitly"
)]
pub fn fixed_bin_histogram2d(
    x: &[f32],
    y: &[f32],
    x_min: f32,
    x_inv: f32,
    y_min: f32,
    y_inv: f32,
    bins: u32,
    hist_x: &mut [u32],
    hist_y: &mut [u32],
    joint: &mut [u32],
) {
    assert_eq!(x.len(), y.len());
    let bins_usize = bins as usize;
    assert!(bins_usize > 0);
    assert!(hist_x.len() >= bins_usize);
    assert!(hist_y.len() >= bins_usize);
    assert!(joint.len() >= bins_usize * bins_usize);

    hist_x[..bins_usize].fill(0);
    hist_y[..bins_usize].fill(0);
    joint[..(bins_usize * bins_usize)].fill(0);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2 is available;
            // input/output slice lengths are checked above and the function does
            // not retain pointers after returning.
            unsafe {
                fixed_bin_histogram2d_avx2(
                    x, y, x_min, x_inv, y_min, y_inv, bins, hist_x, hist_y, joint,
                );
            }
            return;
        }
    }
    fixed_bin_histogram2d_scalar(
        x, y, x_min, x_inv, y_min, y_inv, bins, hist_x, hist_y, joint,
    );
}

#[allow(
    clippy::too_many_arguments,
    reason = "the scalar parity path mirrors the explicit hot-path histogram inputs"
)]
fn fixed_bin_histogram2d_scalar(
    x: &[f32],
    y: &[f32],
    x_min: f32,
    x_inv: f32,
    y_min: f32,
    y_inv: f32,
    bins: u32,
    hist_x: &mut [u32],
    hist_y: &mut [u32],
    joint: &mut [u32],
) {
    let max_bin = bins.saturating_sub(1);
    let bins_usize = bins as usize;
    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_bin = fixed_bin_index(x_value, x_min, x_inv, max_bin) as usize;
        let y_bin = fixed_bin_index(y_value, y_min, y_inv, max_bin) as usize;
        hist_x[x_bin] += 1;
        hist_y[y_bin] += 1;
        joint[x_bin * bins_usize + y_bin] += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    reason = "the AVX2 path mirrors the explicit scalar histogram inputs"
)]
unsafe fn fixed_bin_histogram2d_avx2(
    x: &[f32],
    y: &[f32],
    x_min: f32,
    x_inv: f32,
    y_min: f32,
    y_inv: f32,
    bins: u32,
    hist_x: &mut [u32],
    hist_y: &mut [u32],
    joint: &mut [u32],
) {
    use std::arch::x86_64::*;

    let n = x.len();
    let max_bin = bins.saturating_sub(1);
    let bins_usize = bins as usize;
    let x_min_v = _mm256_set1_ps(x_min);
    let x_inv_v = _mm256_set1_ps(x_inv);
    let y_min_v = _mm256_set1_ps(y_min);
    let y_inv_v = _mm256_set1_ps(y_inv);
    let mut x_scaled_lanes = [0.0f32; 8];
    let mut y_scaled_lanes = [0.0f32; 8];

    let chunks = n / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let x_values = _mm256_loadu_ps(x.as_ptr().add(offset));
        let y_values = _mm256_loadu_ps(y.as_ptr().add(offset));
        let x_scaled = _mm256_mul_ps(_mm256_sub_ps(x_values, x_min_v), x_inv_v);
        let y_scaled = _mm256_mul_ps(_mm256_sub_ps(y_values, y_min_v), y_inv_v);
        _mm256_storeu_ps(x_scaled_lanes.as_mut_ptr(), x_scaled);
        _mm256_storeu_ps(y_scaled_lanes.as_mut_ptr(), y_scaled);
        for lane in 0..8 {
            let x_bin = fixed_bin_from_scaled(x_scaled_lanes[lane], max_bin) as usize;
            let y_bin = fixed_bin_from_scaled(y_scaled_lanes[lane], max_bin) as usize;
            hist_x[x_bin] += 1;
            hist_y[y_bin] += 1;
            joint[x_bin * bins_usize + y_bin] += 1;
        }
    }
    fixed_bin_histogram2d_scalar(
        &x[(chunks * 8)..],
        &y[(chunks * 8)..],
        x_min,
        x_inv,
        y_min,
        y_inv,
        bins,
        hist_x,
        hist_y,
        joint,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_bin_indices_simd_matches_scalar_and_clamps() {
        let values: Vec<f32> = (0..37).map(|i| i as f32 * 0.37 - 2.0).collect();
        let (min, max, bins) = (-2.0f32, 11.0f32, 8u32);
        let inv = bins as f32 / (max - min);
        let simd = fixed_bin_indices(&values, min, inv, bins);
        let scalar = fixed_bin_indices_scalar(&values, min, inv, bins);
        assert_eq!(simd, scalar);
        assert!(simd.iter().all(|&bin| bin < bins));
    }

    #[test]
    fn fixed_bin_indices_define_extreme_f32_overflow_like_device_kernels() {
        let wide = [-f32::MAX, -1.0, -0.0, 0.0, 1.0, f32::MAX];
        let wide_inv = 8.0f32 / (f32::MAX - (-f32::MAX));
        let wide_scalar = fixed_bin_indices_scalar(&wide, -f32::MAX, wide_inv, 8);
        assert_eq!(wide_scalar, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(
            fixed_bin_indices(&wide, -f32::MAX, wide_inv, 8),
            wide_scalar
        );

        let subnormal: Vec<f32> = (0..=8).map(f32::from_bits).collect();
        let subnormal_inv = 8.0f32 / f32::from_bits(8);
        let subnormal_scalar = fixed_bin_indices_scalar(&subnormal, 0.0, subnormal_inv, 8);
        assert_eq!(subnormal_scalar, vec![0, 7, 7, 7, 7, 7, 7, 7, 7]);
        assert_eq!(
            fixed_bin_indices(&subnormal, 0.0, subnormal_inv, 8),
            subnormal_scalar
        );
    }

    #[test]
    fn fixed_bin_histogram2d_matches_index_path() {
        let x: Vec<f32> = (0..73)
            .map(|i| {
                let value = i as f32;
                (value * 0.17).sin() * 3.0 + value * 0.025
            })
            .collect();
        let y: Vec<f32> = (0..73)
            .map(|i| {
                let value = i as f32 + 0.5;
                (value * 0.11).cos() * 2.0 - value * 0.015
            })
            .collect();
        let bins = 24u32;
        let x_min = x.iter().copied().fold(f32::INFINITY, f32::min);
        let x_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let y_min = y.iter().copied().fold(f32::INFINITY, f32::min);
        let y_max = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let x_inv = bins as f32 / (x_max - x_min);
        let y_inv = bins as f32 / (y_max - y_min);
        let x_bins = fixed_bin_indices(&x, x_min, x_inv, bins);
        let y_bins = fixed_bin_indices(&y, y_min, y_inv, bins);

        let mut expected_x = vec![0u32; bins as usize];
        let mut expected_y = vec![0u32; bins as usize];
        let mut expected_joint = vec![0u32; bins as usize * bins as usize];
        for (&x_bin, &y_bin) in x_bins.iter().zip(&y_bins) {
            let x_index = x_bin as usize;
            let y_index = y_bin as usize;
            expected_x[x_index] += 1;
            expected_y[y_index] += 1;
            expected_joint[x_index * bins as usize + y_index] += 1;
        }

        let mut hist_x = vec![u32::MAX; bins as usize];
        let mut hist_y = vec![u32::MAX; bins as usize];
        let mut joint = vec![u32::MAX; bins as usize * bins as usize];
        fixed_bin_histogram2d(
            &x,
            &y,
            x_min,
            x_inv,
            y_min,
            y_inv,
            bins,
            &mut hist_x,
            &mut hist_y,
            &mut joint,
        );

        assert_eq!(hist_x, expected_x);
        assert_eq!(hist_y, expected_y);
        assert_eq!(joint, expected_joint);
    }

    #[test]
    fn fixed_bin_histogram2d_matches_scalar_for_extreme_finite_ranges() {
        let x_pattern = [
            -f32::MAX,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
        ];
        let x: Vec<f32> = (0..41)
            .map(|index| x_pattern[index % x_pattern.len()])
            .collect();
        let y: Vec<f32> = (0..x.len())
            .map(|index| f32::from_bits((index % 9) as u32))
            .collect();
        let bins = 8u32;
        let x_inv = bins as f32 / (f32::MAX - (-f32::MAX));
        let y_inv = bins as f32 / f32::from_bits(8);

        let mut scalar_x = vec![0u32; bins as usize];
        let mut scalar_y = vec![0u32; bins as usize];
        let mut scalar_joint = vec![0u32; bins as usize * bins as usize];
        fixed_bin_histogram2d_scalar(
            &x,
            &y,
            -f32::MAX,
            x_inv,
            0.0,
            y_inv,
            bins,
            &mut scalar_x,
            &mut scalar_y,
            &mut scalar_joint,
        );

        let mut dispatch_x = vec![u32::MAX; bins as usize];
        let mut dispatch_y = vec![u32::MAX; bins as usize];
        let mut dispatch_joint = vec![u32::MAX; bins as usize * bins as usize];
        fixed_bin_histogram2d(
            &x,
            &y,
            -f32::MAX,
            x_inv,
            0.0,
            y_inv,
            bins,
            &mut dispatch_x,
            &mut dispatch_y,
            &mut dispatch_joint,
        );

        assert_eq!(dispatch_x, scalar_x);
        assert_eq!(dispatch_y, scalar_y);
        assert_eq!(dispatch_joint, scalar_joint);
    }
}
