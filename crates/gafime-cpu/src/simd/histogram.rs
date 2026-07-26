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
/// map to zero.
pub fn fixed_bin_indices(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    let mut out = vec![0u32; values.len()];
    fixed_bin_indices_into(values, min, inv, bins, &mut out);
    out
}

/// Writes fixed equal-width bin indices into caller-owned storage.
///
/// This form avoids per-call allocation while preserving the exact scalar
/// mapping documented by [`fixed_bin_indices`].
pub fn fixed_bin_indices_into(values: &[f32], min: f32, inv: f32, bins: u32, out: &mut [u32]) {
    assert_eq!(values.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && bins.saturating_sub(1) <= i32::MAX as u32 {
            // SAFETY: runtime feature detection guarantees AVX2 is available;
            // the implementation reads and writes only complete in-bounds lanes.
            unsafe {
                fixed_bin_indices_avx2(values, min, inv, bins, out);
            }
            return;
        }
    }
    fixed_bin_indices_scalar_into(values, min, inv, bins, out);
}

#[cfg(test)]
fn fixed_bin_indices_scalar(values: &[f32], min: f32, inv: f32, bins: u32) -> Vec<u32> {
    let mut out = vec![0u32; values.len()];
    fixed_bin_indices_scalar_into(values, min, inv, bins, &mut out);
    out
}

fn fixed_bin_indices_scalar_into(values: &[f32], min: f32, inv: f32, bins: u32, out: &mut [u32]) {
    assert_eq!(values.len(), out.len());
    let max_bin = bins.saturating_sub(1);
    for (&value, bin) in values.iter().zip(out) {
        *bin = fixed_bin_index(value, min, inv, max_bin);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fixed_bin_indices_avx2(values: &[f32], min: f32, inv: f32, bins: u32, out: &mut [u32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(values.len(), out.len());
    let n = values.len();
    let max_bin = bins.saturating_sub(1);
    debug_assert!(max_bin <= i32::MAX as u32);
    let min_v = _mm256_set1_ps(min);
    let inv_v = _mm256_set1_ps(inv);

    let chunks = n / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let values_v = _mm256_loadu_ps(values.as_ptr().add(offset));
        let scaled = _mm256_mul_ps(_mm256_sub_ps(values_v, min_v), inv_v);
        let bin_indices = fixed_bins_from_scaled_avx2(scaled, max_bin);
        _mm256_storeu_si256(out.as_mut_ptr().add(offset).cast::<__m256i>(), bin_indices);
    }
    for index in (chunks * 8)..n {
        out[index] = fixed_bin_index(values[index], min, inv, max_bin);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fixed_bins_from_scaled_avx2(
    scaled: std::arch::x86_64::__m256,
    max_bin: u32,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    debug_assert!(max_bin <= i32::MAX as u32);
    let zero_i = _mm256_setzero_si256();
    let max_bin_i = _mm256_set1_epi32(max_bin as i32);
    let max_bin_f = _mm256_set1_ps(max_bin as f32);
    let truncated = _mm256_cvttps_epi32(scaled);
    let lower_clamped = _mm256_max_epi32(truncated, zero_i);
    let clamped = _mm256_min_epi32(lower_clamped, max_bin_i);
    let at_or_above_max = _mm256_cmp_ps(scaled, max_bin_f, _CMP_GE_OQ);
    _mm256_blendv_epi8(clamped, max_bin_i, _mm256_castps_si256(at_or_above_max))
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
        if is_x86_feature_detected!("avx2") && bins.saturating_sub(1) <= i32::MAX as u32 {
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
    let mut x_bin_lanes = [0u32; 8];
    let mut y_bin_lanes = [0u32; 8];

    let chunks = n / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let x_values = _mm256_loadu_ps(x.as_ptr().add(offset));
        let y_values = _mm256_loadu_ps(y.as_ptr().add(offset));
        let x_scaled = _mm256_mul_ps(_mm256_sub_ps(x_values, x_min_v), x_inv_v);
        let y_scaled = _mm256_mul_ps(_mm256_sub_ps(y_values, y_min_v), y_inv_v);
        let x_bins = fixed_bins_from_scaled_avx2(x_scaled, max_bin);
        let y_bins = fixed_bins_from_scaled_avx2(y_scaled, max_bin);
        _mm256_storeu_si256(x_bin_lanes.as_mut_ptr().cast::<__m256i>(), x_bins);
        _mm256_storeu_si256(y_bin_lanes.as_mut_ptr().cast::<__m256i>(), y_bins);
        for lane in 0..8 {
            let x_bin = x_bin_lanes[lane] as usize;
            let y_bin = y_bin_lanes[lane] as usize;
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
    #[ignore = "manual release benchmark; run with --release --ignored --nocapture"]
    fn fixed_bin_release_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        fn median_ns(mut samples: Vec<u128>) -> u128 {
            samples.sort_unstable();
            samples[samples.len() / 2]
        }

        let rows = 1_048_576usize;
        let bins = 96u32;
        let warmups = 20usize;
        let repetitions = 101usize;
        let x: Vec<f32> = (0..rows)
            .map(|index| ((index.wrapping_mul(37) % 100_003) as f32) / 100_003.0)
            .collect();
        let y: Vec<f32> = (0..rows)
            .map(|index| ((index.wrapping_mul(91) % 100_019) as f32) / 100_019.0)
            .collect();
        let inv = bins as f32;
        let mut hist_x = vec![0u32; bins as usize];
        let mut hist_y = vec![0u32; bins as usize];
        let mut joint = vec![0u32; bins as usize * bins as usize];

        for _ in 0..warmups {
            fixed_bin_histogram2d(
                &x,
                &y,
                0.0,
                inv,
                0.0,
                inv,
                bins,
                &mut hist_x,
                &mut hist_y,
                &mut joint,
            );
            black_box(fixed_bin_indices(&x, 0.0, inv, bins));
        }

        let histogram_samples = (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                fixed_bin_histogram2d(
                    &x,
                    &y,
                    0.0,
                    inv,
                    0.0,
                    inv,
                    bins,
                    &mut hist_x,
                    &mut hist_y,
                    &mut joint,
                );
                black_box((&hist_x, &hist_y, &joint));
                start.elapsed().as_nanos()
            })
            .collect();
        let allocated_index_samples = (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                black_box(fixed_bin_indices(&x, 0.0, inv, bins));
                start.elapsed().as_nanos()
            })
            .collect();
        let mut reusable_indices = vec![0u32; rows];
        for _ in 0..warmups {
            fixed_bin_indices_into(&x, 0.0, inv, bins, &mut reusable_indices);
            black_box(&reusable_indices);
        }
        let reusable_index_samples = (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                fixed_bin_indices_into(&x, 0.0, inv, bins, &mut reusable_indices);
                black_box(&reusable_indices);
                start.elapsed().as_nanos()
            })
            .collect();

        println!(
            "rows={rows} bins={bins} samples={repetitions} histogram_median_ns={} allocated_indices_median_ns={} reusable_indices_median_ns={}",
            median_ns(histogram_samples),
            median_ns(allocated_index_samples),
            median_ns(reusable_index_samples),
        );
    }

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
    fn fixed_bin_indices_match_scalar_for_adversarial_lanes_tails_and_bin_ladder() {
        const BIN_LADDER: [u32; 10] = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96];
        const LENGTHS: [usize; 4] = [8, 17, 1_000, 4_096];

        for bins in BIN_LADDER {
            let max_bin = bins.saturating_sub(1) as f32;
            let below_max = f32::from_bits(max_bin.to_bits().saturating_sub(1));
            let above_max = f32::from_bits(max_bin.to_bits().saturating_add(1));
            let pattern = [
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -0.0,
                0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                -1.0,
                0.999_999_94,
                1.0,
                below_max,
                max_bin,
                above_max,
                f32::MAX,
                -f32::MAX,
            ];
            for length in LENGTHS {
                let values: Vec<f32> = (0..length)
                    .map(|index| pattern[index % pattern.len()])
                    .collect();
                let expected = fixed_bin_indices_scalar(&values, 0.0, 1.0, bins);
                let mut reusable = vec![u32::MAX; length];
                fixed_bin_indices_into(&values, 0.0, 1.0, bins, &mut reusable);
                assert_eq!(reusable, expected, "bins={bins} length={length}");
                assert_eq!(
                    fixed_bin_indices(&values, 0.0, 1.0, bins),
                    expected,
                    "allocated bins={bins} length={length}"
                );

                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let mut direct = vec![u32::MAX; length];
                    // SAFETY: the runtime check proves AVX2 support. The output
                    // has exactly the input length and this ladder fits i32.
                    unsafe {
                        fixed_bin_indices_avx2(&values, 0.0, 1.0, bins, &mut direct);
                    }
                    assert_eq!(direct, expected, "AVX2 bins={bins} length={length}");
                }
            }
        }
    }

    #[test]
    fn fixed_bin_indices_large_bin_domain_uses_exact_scalar_fallback() {
        let bins = i32::MAX as u32 + 2;
        let values = [
            f32::NAN,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.0,
            1.0e20,
            f32::INFINITY,
        ];
        let expected = fixed_bin_indices_scalar(&values, 0.0, 1.0, bins);
        let mut actual = vec![u32::MAX; values.len()];
        fixed_bin_indices_into(&values, 0.0, 1.0, bins, &mut actual);
        assert_eq!(actual, expected);
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

    #[test]
    fn fixed_bin_histogram2d_matches_scalar_for_adversarial_ladder_and_tails() {
        const BIN_LADDER: [u32; 10] = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96];
        const LENGTHS: [usize; 4] = [8, 17, 1_000, 4_096];

        for bins in BIN_LADDER {
            let max_bin = bins.saturating_sub(1) as f32;
            let below_max = f32::from_bits(max_bin.to_bits().saturating_sub(1));
            let above_max = f32::from_bits(max_bin.to_bits().saturating_add(1));
            let pattern = [
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -0.0,
                0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                -1.0,
                0.999_999_94,
                1.0,
                below_max,
                max_bin,
                above_max,
                f32::MAX,
                -f32::MAX,
            ];
            for length in LENGTHS {
                let x: Vec<f32> = (0..length)
                    .map(|index| pattern[index % pattern.len()])
                    .collect();
                let y: Vec<f32> = (0..length)
                    .map(|index| pattern[(pattern.len() - 1) - (index % pattern.len())])
                    .collect();
                let histogram_len = bins as usize;
                let joint_len = histogram_len * histogram_len;
                let mut expected_x = vec![0u32; histogram_len];
                let mut expected_y = vec![0u32; histogram_len];
                let mut expected_joint = vec![0u32; joint_len];
                fixed_bin_histogram2d_scalar(
                    &x,
                    &y,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    bins,
                    &mut expected_x,
                    &mut expected_y,
                    &mut expected_joint,
                );

                let mut actual_x = vec![u32::MAX; histogram_len];
                let mut actual_y = vec![u32::MAX; histogram_len];
                let mut actual_joint = vec![u32::MAX; joint_len];
                fixed_bin_histogram2d(
                    &x,
                    &y,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    bins,
                    &mut actual_x,
                    &mut actual_y,
                    &mut actual_joint,
                );
                assert_eq!(actual_x, expected_x, "x bins={bins} length={length}");
                assert_eq!(actual_y, expected_y, "y bins={bins} length={length}");
                assert_eq!(
                    actual_joint, expected_joint,
                    "joint bins={bins} length={length}"
                );
                assert_eq!(actual_x.iter().sum::<u32>(), length as u32);
                assert_eq!(actual_y.iter().sum::<u32>(), length as u32);
                assert_eq!(actual_joint.iter().sum::<u32>(), length as u32);
            }
        }
    }
}
