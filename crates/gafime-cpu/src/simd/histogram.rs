/// Fixed equal-width bin indices for `values`: `bin = clamp(trunc((v - min) * inv),
/// 0, bins-1)` — the same mapping the GPU MI kernel uses. SIMD (AVX2, 8-wide) with
/// a scalar fallback. Non-finite values are excluded upstream by the finite-pair
/// filter, so their exact bin is irrelevant.
pub fn fixed_bin_indices(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fixed_bin_indices_avx2(values, min, inv, bins) };
        }
    }
    fixed_bin_indices_scalar(values, min, inv, bins)
}

fn fixed_bin_indices_scalar(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    let max_bin = bins.saturating_sub(1);
    values
        .iter()
        .map(|&v| (((v - min) * inv) as u32).min(max_bin))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fixed_bin_indices_avx2(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    use std::arch::x86_64::*;

    let n = values.len();
    let mut out = vec![0u32; n];
    let max_bin = bins.saturating_sub(1) as i32;
    let min_v = _mm256_set1_ps(min);
    let inv_v = _mm256_set1_ps(inv);
    let zero = _mm256_setzero_si256();
    let max_v = _mm256_set1_epi32(max_bin);

    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        let v = _mm256_loadu_ps(values.as_ptr().add(o));
        let scaled = _mm256_mul_ps(_mm256_sub_ps(v, min_v), inv_v);
        // truncate toward zero -> i32, then clamp into [0, bins-1]
        let mut idx = _mm256_cvttps_epi32(scaled);
        idx = _mm256_max_epi32(idx, zero);
        idx = _mm256_min_epi32(idx, max_v);
        _mm256_storeu_si256(out.as_mut_ptr().add(o) as *mut __m256i, idx);
    }
    for i in (chunks * 8)..n {
        out[i] = (((values[i] - min) * inv) as i32).clamp(0, max_bin) as u32;
    }
    out
}

/// Build fixed-bin marginal and joint histograms for the GPU-compatible MI
/// approximation. AVX2 vectorizes the bin arithmetic and feeds lane-local scalar
/// histogram increments; the scatter itself is data-dependent and remains scalar.
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
    let max_bin = bins.saturating_sub(1) as usize;
    let bins_usize = bins as usize;
    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_bin = (((x_value - x_min) * x_inv) as usize).min(max_bin);
        let y_bin = (((y_value - y_min) * y_inv) as usize).min(max_bin);
        hist_x[x_bin] += 1;
        hist_y[y_bin] += 1;
        joint[x_bin * bins_usize + y_bin] += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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
    let max_bin = bins.saturating_sub(1) as i32;
    let bins_usize = bins as usize;
    let x_min_v = _mm256_set1_ps(x_min);
    let x_inv_v = _mm256_set1_ps(x_inv);
    let y_min_v = _mm256_set1_ps(y_min);
    let y_inv_v = _mm256_set1_ps(y_inv);
    let zero = _mm256_setzero_si256();
    let max_v = _mm256_set1_epi32(max_bin);
    let mut x_bins = [0i32; 8];
    let mut y_bins = [0i32; 8];

    let chunks = n / 8;
    for c in 0..chunks {
        let o = c * 8;
        let x_v = _mm256_loadu_ps(x.as_ptr().add(o));
        let y_v = _mm256_loadu_ps(y.as_ptr().add(o));
        let x_scaled = _mm256_mul_ps(_mm256_sub_ps(x_v, x_min_v), x_inv_v);
        let y_scaled = _mm256_mul_ps(_mm256_sub_ps(y_v, y_min_v), y_inv_v);
        let mut x_idx = _mm256_cvttps_epi32(x_scaled);
        let mut y_idx = _mm256_cvttps_epi32(y_scaled);
        x_idx = _mm256_max_epi32(_mm256_min_epi32(x_idx, max_v), zero);
        y_idx = _mm256_max_epi32(_mm256_min_epi32(y_idx, max_v), zero);
        _mm256_storeu_si256(x_bins.as_mut_ptr() as *mut __m256i, x_idx);
        _mm256_storeu_si256(y_bins.as_mut_ptr() as *mut __m256i, y_idx);
        for lane in 0..8 {
            let x_bin = x_bins[lane] as usize;
            let y_bin = y_bins[lane] as usize;
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
        assert!(simd.iter().all(|&b| b < bins));
    }

    #[test]
    fn fixed_bin_histogram2d_matches_index_path() {
        let x: Vec<f32> = (0..73)
            .map(|i| {
                let v = i as f32;
                (v * 0.17).sin() * 3.0 + v * 0.025
            })
            .collect();
        let y: Vec<f32> = (0..73)
            .map(|i| {
                let v = i as f32 + 0.5;
                (v * 0.11).cos() * 2.0 - v * 0.015
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
            let a = x_bin as usize;
            let b = y_bin as usize;
            expected_x[a] += 1;
            expected_y[b] += 1;
            expected_joint[a * bins as usize + b] += 1;
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
}
