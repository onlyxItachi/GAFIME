#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]


def source_checks() -> dict[str, bool]:
    simd_root = ROOT / "crates/gafime-cpu/src/simd"
    dispatch = (ROOT / "crates/gafime-cpu/src/dispatch.rs").read_text()
    isa = (simd_root / "isa.rs").read_text()
    simd_mod = (simd_root / "mod.rs").read_text()
    covariance = (simd_root / "covariance.rs").read_text()
    histogram = (simd_root / "histogram.rs").read_text()
    histogram_avx2 = histogram.split(
        "unsafe fn fixed_bin_histogram2d_avx2", 1
    )[1].split("#[cfg(test)]", 1)[0]
    matrix = (ROOT / "crates/gafime-cpu/src/matrix.rs").read_text()
    kernels = (ROOT / "crates/gafime-cpu/src/kernels/mod.rs").read_text()
    checks = {
        "dispatch_is_compat_reexport": "pub use crate::simd::*" in dispatch,
        "simd_dispatch_declared": all(
            needle in covariance
            for needle in (
                "pearson_sums_avx512",
                "pearson_sums_avx2",
                "pearson_sums_sse42",
                "pearson_sums_neon",
            )
        )
        and "finite_dispatch_isa" in isa,
        "centered_covariance_kept": "self.sxy / denom" in covariance
        and "clamp(-1.0, 1.0)" in covariance,
        "finite_check_fused_into_sum_pass": "all_pairs_finite" not in covariance
        and "EARLY_NONFINITE_PROBE_ROWS: usize = 16" in covariance
        and "all_finite_avx512_pd" in covariance
        and "all_finite_avx2_pd" in covariance
        and "all_finite_neon_f64" in covariance,
        "column_native_matrix": "columns: Vec<f32>" in matrix
        and "transpose_row_major" in matrix,
        "scratch_reuse_scoring": "ContinuousScoreScratch" in kernels
        and "score_continuous_combo_into" in kernels,
        "pearson_r2_share_covariance": "cached_pearson" in kernels
        and "get_or_insert_with(|| pearson(signal, matrix.target()))" in kernels,
        "arity1_uses_column_slice": "matrix.column(combo[0] as usize)" in kernels,
        "no_row_vector_allocation_in_interactions": "Vec::with_capacity(rows)" not in kernels,
        "fixed_bin_mi_histogram_simd": "fixed_bin_histogram2d" in histogram
        and "fixed_bin_histogram2d_avx2" in histogram
        and "simd::fixed_bin_histogram2d" in kernels
        and "simd::fixed_bin_indices(&x_values" not in kernels,
        "branchless_fixed_bin_conversion": "fixed_bins_from_scaled_avx2" in histogram
        and "_mm256_cvttps_epi32" in histogram
        and "_mm256_blendv_epi8" in histogram
        and "scaled_lanes" not in histogram,
        "safe_histogram_scatter": "hist_x[x_bin] += 1" in histogram_avx2
        and "hist_y[y_bin] += 1" in histogram_avx2
        and "joint[x_bin * bins_usize + y_bin] += 1" in histogram_avx2
        and "get_unchecked" not in histogram_avx2,
        "reusable_fixed_bin_output": "pub fn fixed_bin_indices_into" in histogram
        and "fixed_bin_indices_into" in simd_mod
        and "_mm256_storeu_si256(out.as_mut_ptr()" in histogram,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"CPU native kernel source checks failed: {failed}")
    return checks


def timed_check(name: str, command: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return {
        "name": name,
        "seconds": round(time.perf_counter() - start, 6),
        "command": command,
    }


def main() -> None:
    checks = source_checks()
    runs = [
        timed_check("simd_tests", ["cargo", "test", "-p", "gafime-cpu", "simd::"]),
        timed_check("matrix_tests", ["cargo", "test", "-p", "gafime-cpu", "matrix::tests"]),
        timed_check(
            "continuous_backend_tests",
            ["cargo", "test", "-p", "gafime-cpu", "tests::cpu_backend_executes"],
        ),
    ]
    print(json.dumps({"checks": checks, "runs": runs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
