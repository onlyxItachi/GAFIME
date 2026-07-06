#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import types


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_RUNTIME_MODULES = {
    "gafime.backends",
    "gafime.backends.core_backend",
    "gafime.backends.native_cuda_backend",
    "gafime.backends.native_rocm_backend",
    "gafime.compile.scenario",
    "gafime.compile.sessions",
    "gafime.engine",
    "gafime.native_data",
}
FORBIDDEN_RUNTIME_STRINGS = (
    "GAFIME_V1_ENGINE",
    "GAFIME_USE_LEGACY_ENGINE",
    "gafime_core",
    "native_cuda_backend",
    "native_rocm_backend",
    "compile.sessions",
    "compile.scenario",
)
FORBIDDEN_LOCAL_RUNTIME_PATHS = (
    "gafime",
    "gafime_core",
    "gafime.egg-info",
    "src/cpu",
    "tools",
    "python/gafime/backends",
    "python/gafime/metrics",
    "python/gafime/native_data.py",
    "python/gafime/planning",
    "python/gafime/validation",
)
FORBIDDEN_LOCAL_RUNTIME_GLOBS = (
    "gafime/_native*.so",
    "gafime/gafime_core*.so",
    "gafime/gafime_cpu*.so",
    "python/gafime/_native*.so",
    "python/gafime/gafime_core*.so",
    "python/gafime/gafime_cpu*.so",
)


class FakeRecord:
    def __init__(self, combo, metrics, candidate_id):
        self.combo = combo
        self.metrics = metrics
        self.candidate_id = candidate_id


class FakeReport:
    rows = 4
    cols = 2
    max_arity = 1
    metric_ids = [1, 4]

    def __init__(self, length=2):
        self.length = int(length)

    def __len__(self):
        return self.length

    def record(self, index):
        return FakeRecord(self.combo(index), self.metric_values(index), self.candidate_id(index))

    def combo(self, index):
        return [int(index % 2)]

    def metric_values(self, index):
        return [1.0 if index % 2 == 0 else -1.0, 1.0]

    def candidate_id(self, index):
        return int(index)

    def ranked_indices(self, *, metric_index=None, descending=True, limit=None):
        count = self.length if limit is None else min(int(limit), self.length)
        return list(range(count))

    def records(self):
        raise AssertionError("v1 gate forbids normal-path Python report list materialization")


class FakeArtifact:
    backend_name = "v1-rust-cpu"
    device = "cpu"
    is_gpu = False

    def __init__(self, calls, length=2):
        self.calls = calls
        self.closed = False
        self.length = length

    def analyze(self):
        return FakeReport(self.length)

    def close(self):
        self.closed = True


def install_fake_boundary(length=2):
    module = types.ModuleType("_v1_gate_fake_boundary")
    calls = []

    def compile_continuous(config, features, target, *, rows, cols):
        calls.append(
            {
                "config": config,
                "features": features,
                "target": target,
                "rows": rows,
                "cols": cols,
            }
        )
        return FakeArtifact(calls, length=length)

    module.compile_continuous = compile_continuous
    module.BOUNDARY_NAME = "fake-gafime-py"
    module.calls = calls
    sys.modules[module.__name__] = module
    os.environ["GAFIME_V1_BOUNDARY_MODULE"] = module.__name__
    return module


def check_packaging() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["build-system"]["build-backend"] == "maturin"
    assert pyproject["tool"]["maturin"]["manifest-path"] == "crates/gafime-py/Cargo.toml"
    assert pyproject["tool"]["maturin"]["module-name"] == "gafime.gafime_py"
    assert pyproject["tool"]["maturin"]["python-source"] == "python"
    build_requires = " ".join(pyproject["build-system"].get("requires", []))
    assert "setuptools" not in build_requires
    assert "pybind11" not in build_requires
    assert "cmake" not in build_requires

    setup_text = (ROOT / "setup.py").read_text()
    assert "no longer builds runtime artifacts" in setup_text
    manifest_text = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include gafime_core" not in manifest_text
    assert "RustCompileSrc" not in manifest_text

    packaged_files = {path.relative_to(ROOT / "python").as_posix() for path in (ROOT / "python").rglob("*") if path.is_file()}
    forbidden_paths = {
        "gafime/backends",
        "gafime/native_data.py",
        "gafime/compile/sessions.py",
        "gafime/compile/scenario.py",
    }
    for forbidden in forbidden_paths:
        assert all(not item.startswith(forbidden) for item in packaged_files), forbidden

    package_text = "\n".join(path.read_text() for path in (ROOT / "python" / "gafime").rglob("*.py"))
    for forbidden in FORBIDDEN_RUNTIME_STRINGS:
        assert forbidden not in package_text, forbidden


def check_no_local_legacy_runtime_artifacts() -> None:
    offenders = []
    for relative in FORBIDDEN_LOCAL_RUNTIME_PATHS:
        path = ROOT / relative
        if path.exists():
            offenders.append(relative)
    for pattern in FORBIDDEN_LOCAL_RUNTIME_GLOBS:
        offenders.extend(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
    assert not offenders, f"legacy local runtime artifacts are present: {sorted(offenders)}"


def check_runtime_surface() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    fake = install_fake_boundary()
    os.environ["GAFIME_USE_LEGACY_ENGINE"] = "1"
    try:
        import gafime

        before_modules = set(sys.modules)
        cfg = gafime.EngineConfig(
            backend="core",
            metric_names=("pearson", "r2"),
            budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
            permutation_tests=0,
            num_repeats=1,
        )
        report = gafime.GafimeEngine(cfg).analyze(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
            [1.0, 2.0, 3.0, 4.0],
            ["a", "b"],
        )
        families = {family.name: family for family in gafime.available_families()}
    finally:
        os.environ.pop("GAFIME_USE_LEGACY_ENGINE", None)

    assert fake.calls, "public analyze did not use the v1 boundary"
    assert fake.calls[0]["features"] == [1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]
    assert report.backend.name == "v1-rust-cpu"
    assert report.interactions.is_native_backed
    assert report.interactions[0].metrics == {"pearson": 1.0, "r2": 1.0}
    assert report.interactions.top_k(1)[0].combo == (0,)
    assert families["continuous"].supported
    assert families["continuous"].metal_kernel
    assert families["decision_path"].supported
    assert families["decision_path"].metal_kernel
    assert families["time_series"].supported
    assert families["time_series"].metal_kernel
    assert all(not family.python_candidate_loop for family in families.values())
    loaded_forbidden = FORBIDDEN_RUNTIME_MODULES.intersection(set(sys.modules) - before_modules)
    assert not loaded_forbidden, sorted(loaded_forbidden)


def check_no_source_opt_in_or_fallback() -> None:
    source_text = "\n".join(path.read_text() for path in (ROOT / "python" / "gafime").rglob("*.py"))
    assert "GAFIME_V1_ENGINE" not in source_text
    assert "GAFIME_USE_LEGACY_ENGINE" not in source_text


def check_native_kernel_structure() -> None:
    simd_root = ROOT / "crates" / "gafime-cpu" / "src" / "simd"
    simd_mod_text = (simd_root / "mod.rs").read_text()
    isa_text = (simd_root / "isa.rs").read_text()
    covariance_text = (simd_root / "covariance.rs").read_text()
    histogram_text = (simd_root / "histogram.rs").read_text()
    dispatch_text = (ROOT / "crates" / "gafime-cpu" / "src" / "dispatch.rs").read_text()
    assert "pub use crate::simd::*" in dispatch_text
    assert "mod covariance" in simd_mod_text
    assert "mod histogram" in simd_mod_text
    assert "mod isa" in simd_mod_text
    assert "finite_dispatch_isa" in isa_text
    assert "pearson_sums_avx2" in covariance_text
    assert "pearson_sums_sse42" in covariance_text
    assert "pearson_sums_neon" in covariance_text
    assert "fixed_bin_histogram2d" in histogram_text
    assert "fixed_bin_histogram2d_avx2" in histogram_text
    assert "#[target_feature(enable = \"avx2\")]" in covariance_text
    assert "#[target_feature(enable = \"sse4.2\")]" in covariance_text
    assert "#[target_feature(enable = \"avx2\")]" in histogram_text
    finite_body = covariance_text.split("fn pearson_sums_finite", 1)[1].split("fn all_pairs_finite", 1)[0]
    assert "pearson_sums_avx2" in finite_body
    assert "pearson_sums_sse42" in finite_body
    assert "pearson_sums_neon" in finite_body

    matrix_text = (ROOT / "crates" / "gafime-cpu" / "src" / "matrix.rs").read_text()
    assert "columns: Vec<f32>" in matrix_text
    assert "transpose_row_major" in matrix_text
    assert "self.columns[col * self.rows as usize + row]" in matrix_text

    kernels_text = (ROOT / "crates" / "gafime-cpu" / "src" / "kernels" / "mod.rs").read_text()
    assert "ContinuousScoreScratch" in kernels_text
    assert "score_continuous_combo_into" in kernels_text
    assert "matrix.column(combo[0] as usize)" in kernels_text
    assert "simd::fixed_bin_histogram2d" in kernels_text
    assert "simd::fixed_bin_indices(&x_values" not in kernels_text
    assert "Vec::with_capacity(rows)" not in kernels_text

    native_root = ROOT / "src"
    rust_gpu_crate = ROOT / "crates" / "gafime-gpu-sys" / "src"
    cuda_root = native_root / "cuda"
    rocm_root = native_root / "rocm"
    metal_root = native_root / "metal"
    common_root = native_root / "common"

    assert not (ROOT / "gpu").exists(), "v1 GPU runtime sources must live under root src/"
    for stale_root in ("common", "cuda", "rocm", "metal"):
        stale_path = rust_gpu_crate / stale_root
        if stale_path.exists():
            assert not any(path.is_file() for path in stale_path.rglob("*")), (
                "crates/gafime-gpu-sys/src must stay Rust-only; native sources belong under root src/"
            )
    assert (common_root / "gafime_gpu_abi.hpp").exists()
    assert (common_root / "gpu_abi_impl.hpp").exists()
    assert (cuda_root / "cuda_api.hpp").exists()
    assert (rocm_root / "rocm_api.hpp").exists()
    assert (metal_root / "metal_api.hpp").exists()

    for backend_root in (cuda_root, rocm_root, metal_root, common_root):
        for path in backend_root.iterdir():
            if path.name == "CMakeLists.txt":
                continue
            if path.name == "embed_ptx.cmake":
                assert backend_root == cuda_root
                continue
            assert path.suffix in {".hpp", ".cuh", ".cu", ".hip", ".metal", ".mm"}, path
            assert path.suffix not in {".h", ".cpp"}, path

    cuda_launcher = (cuda_root / "launcher.cu").read_text()
    cuda_kernels = (cuda_root / "kernels.cu").read_text()
    cuda_header = (cuda_root / "kernels.cuh").read_text()
    cuda_rt_launcher = (cuda_root / "rt_launcher.cu").read_text()
    cuda_rt_kernels = (cuda_root / "rt_kernels.cu").read_text()
    cuda_rt_header = (cuda_root / "rt_kernels.cuh").read_text()
    cuda_rt_launcher_header = (cuda_root / "rt_launcher.cuh").read_text()
    cuda_cmake = (cuda_root / "CMakeLists.txt").read_text()
    rocm_launcher = (rocm_root / "launcher.hip").read_text()
    rocm_kernels = (rocm_root / "kernels.hip").read_text()
    rocm_header = (rocm_root / "kernels.hpp").read_text()
    rocm_cmake = (rocm_root / "CMakeLists.txt").read_text()
    metal_launcher = (metal_root / "launcher.mm").read_text()
    metal_shader = (metal_root / "shader.metal").read_text()
    metal_cmake = (metal_root / "CMakeLists.txt").read_text()
    common_header = (common_root / "gafime_gpu_abi.hpp").read_text()
    contract_workflow = (ROOT / ".github" / "workflows" / "v1_contract_validation.yml").read_text()
    cuda_abi_smoke = (ROOT / "tests" / "gpu" / "cuda_v1_abi_smoke.cpp").read_text()
    optix_smoke = (ROOT / "tests" / "gpu" / "cuda_rt_decision_path_optix_smoke.cu").read_text()
    cuda_rt_scale_bench = (ROOT / "tests" / "gpu" / "cuda_rt_membership_scale_bench.cpp").read_text()

    for name, launcher_text in (("cuda", cuda_launcher), ("rocm", rocm_launcher), ("metal", metal_launcher)):
        assert "__global__" not in launcher_text, f"{name} launcher owns device kernels"
        assert "__device__" not in launcher_text, f"{name} launcher owns device helpers"
        assert "placeholder" not in launcher_text.lower(), name
    assert "__global__" not in cuda_rt_launcher
    assert "__device__" not in cuda_rt_launcher
    assert "placeholder" not in cuda_rt_launcher.lower()

    assert "<<<" in cuda_launcher, "CUDA launcher owns <<<>>> launch calls"
    assert "<<<" in cuda_rt_launcher, "CUDA RT launcher owns RT <<<>>> launch calls"
    assert "<<<" not in cuda_kernels, "CUDA kernels file must not own launches"
    assert "<<<" not in cuda_rt_kernels, "CUDA RT kernels file must not own launches"
    assert "hipLaunchKernelGGL" in rocm_launcher, "ROCm launcher owns HIP launch calls"
    assert "hipLaunchKernelGGL" not in rocm_kernels and "<<<" not in rocm_kernels

    for name, device_text in (("cuda", cuda_kernels), ("rocm", rocm_kernels)):
        assert "__global__ void score_continuous_chunk_kernel" in device_text, name
        assert "__global__ void score_spearman_chunk_kernel" in device_text, name
        assert "__global__ void score_mutual_info_chunk_kernel" in device_text, name
        assert "placeholder" not in device_text.lower(), name
        assert "row * cols + combo" not in device_text, f"{name} kernels must not scan sample-major features"
        assert "row * cols + col" not in device_text, f"{name} kernels must not scan sample-major features"
        assert "row * n_features" not in device_text, f"{name} kernels must not scan sample-major features"
        assert "interaction_value(features, column_means, i, n_features" not in device_text, (
            f"{name} kernels must not pass feature-count as the feature-major stride"
        )
        assert "interaction_value(features, column_means, j, n_features" not in device_text, (
            f"{name} kernels must not pass feature-count as the feature-major stride"
        )
        assert "static_cast<uint64_t>(col) * rows + row" in device_text, (
            f"{name} kernels must read feature-major resident features"
        )
        assert "interaction_value(features, column_means, row, n_samples" in device_text, (
            f"{name} kernels must pass rows as the feature-major stride"
        )
    assert "__global__ void selected_metric_max_kernel" in cuda_kernels
    assert "__global__ void accumulate_exceedances_kernel" in cuda_kernels
    assert "__global__ void decision_path_membership_kernel" not in cuda_kernels
    assert "score_decision_path_direct_stats_kernel" not in cuda_kernels
    assert "direct_inside_counts" not in cuda_kernels
    assert "__global__ void decision_path_membership_kernel" in cuda_rt_kernels
    assert "__raygen__gafime_dp" in cuda_rt_kernels
    assert "__intersection__gafime_dp_box" in cuda_rt_kernels
    assert "__anyhit__gafime_dp_mark" in cuda_rt_kernels
    assert "pack_decision_path_points_kernel" in cuda_rt_kernels
    assert "pack_grouped_decision_path_points_kernel" in cuda_rt_kernels
    assert "scatter_decision_path_score_metrics_kernel" in cuda_rt_kernels
    assert "rt_kernel::decision_path_membership_kernel" in cuda_rt_launcher
    assert "optixLaunch" in cuda_rt_launcher
    assert "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES" in cuda_rt_launcher
    assert "OPTIX_BUILD_INPUT_TYPE_TRIANGLES" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_GEOMETRY" in cuda_rt_launcher
    assert "rt_plan_signature" in cuda_rt_launcher
    assert "execute_decision_path_score" in cuda_rt_launcher
    assert "score_decision_path_bitset_kernel" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" in cuda_rt_launcher
    assert "build_rt_score_groups" in cuda_rt_launcher
    assert "execute_decision_path_score_optix_grouped" in cuda_rt_launcher
    assert "execute_decision_path_score_optix_grouped_instanced" in cuda_rt_launcher
    assert "RtGeometryMode::Triangle2dInstanced" in cuda_rt_launcher
    assert "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING" in cuda_rt_launcher
    assert "OPTIX_BUILD_INPUT_TYPE_INSTANCES" in cuda_rt_launcher
    assert "rt_instanced_group_signature" in cuda_rt_launcher
    assert "bool rebuild_geometry" in cuda_rt_launcher
    assert "program.gas_signature = geometry_signature" in cuda_rt_launcher
    assert "params.handle = program.gas_handle" in cuda_rt_launcher
    assert "point_group_stride" in cuda_rt_launcher
    assert "packed_points_valid" in cuda_rt_launcher
    assert "packed_points_generation" in cuda_rt_launcher
    assert "feature_generation" in cuda_launcher
    assert "feature_generation" in cuda_rt_launcher
    assert "target_generation" in cuda_launcher
    assert "target_generation" in cuda_rt_launcher
    assert "target_stats_valid" in cuda_rt_launcher
    assert "target_stats_generation" in cuda_rt_launcher
    assert "rt_score_batch_signature" in cuda_rt_launcher
    assert "grouped_score_plan_valid" in cuda_rt_launcher
    assert "grouped_score_plan_signature" in cuda_rt_launcher
    assert "grouped_original_paths_valid" in cuda_rt_launcher
    assert "grouped_original_paths_signature" in cuda_rt_launcher
    assert "grouped_final_metric_values_device" in cuda_rt_launcher
    assert "grouped_original_paths_device" in cuda_rt_launcher
    assert "cudaMemcpyAsync" in cuda_rt_launcher
    assert "group_path_offsets_device" in cuda_rt_launcher
    assert "original_paths" in cuda_rt_launcher
    assert "for (RtScoreGroup& group : groups)" in cuda_rt_launcher
    assert "bool placed = false" in cuda_rt_launcher
    assert "precomputed_target_stats_device" in cuda_rt_launcher
    assert "metric_values_out" in cuda_rt_launcher
    grouped_body = cuda_rt_launcher.split("int execute_decision_path_score_optix_grouped(\n", 1)[1]
    grouped_body = grouped_body.split("int execute_decision_path_score_optix(", 1)[0]
    assert "GafimeResultTable group_result" not in grouped_body
    assert "group_combo_indices" not in grouped_body
    assert "group_metric_values" not in grouped_body
    assert "final_metric_values_device" in grouped_body
    assert "flattened_original_paths" in grouped_body
    assert "group_original_path_offsets" in grouped_body
    assert "group.original_paths.data()" not in grouped_body
    assert "scatter_decision_path_score_metrics_kernel" in cuda_rt_launcher
    assert "shared_target_stats" in cuda_rt_launcher
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_launcher
    assert "decision_path_target_stats_kernel" in cuda_rt_launcher
    assert "direct_inside_counts_device" in cuda_rt_launcher
    assert "direct_inside_sum_y_device" in cuda_rt_launcher
    assert "membership_words_device" in cuda_rt_launcher
    assert "words_per_path" in cuda_rt_launcher
    assert "atomicOr" in cuda_rt_kernels
    assert "atomicAdd(&params.direct_inside_counts" in cuda_rt_kernels
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_kernels
    assert "decision_path_target_stats_kernel" in cuda_rt_kernels
    assert "execute_decision_path_membership_sm" in cuda_rt_launcher
    assert "features_are_finite" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT" in cuda_rt_launcher
    assert "execute_decision_path_membership" in cuda_rt_launcher
    assert "gafime_gpu_decision_path_membership" not in cuda_rt_launcher
    assert "gafime_gpu_decision_path_score" not in cuda_rt_launcher
    assert "props.major == 8 && props.minor >= 9" in cuda_launcher
    assert "gafime_gpu_permutation_pvalues" in cuda_launcher
    assert "gafime_gpu_decision_path_membership" in cuda_launcher
    assert "gafime_gpu_decision_path_score" in cuda_launcher
    assert "execute_decision_path_membership" in cuda_launcher
    assert "execute_decision_path_score" in cuda_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" not in cuda_launcher
    assert "direct_inside_counts" not in cuda_launcher
    assert "mix_permutation_seed" in cuda_launcher
    assert "0xA5A5A5A5" in cuda_launcher
    assert "--score-only" in cuda_rt_scale_bench
    assert "--mixed-axes" in cuda_rt_scale_bench
    assert "--mixed-axis-pairs=" in cuda_rt_scale_bench
    assert "mixed_axes && !score_only" in cuda_rt_scale_bench
    assert "box.feature0" in cuda_rt_scale_bench
    assert "box.feature1" in cuda_rt_scale_bench
    assert "time_cpu_score_boxes" in cuda_rt_scale_bench
    assert "ScoreResult gpu_rt_scores" in cuda_rt_scale_bench
    assert "std::vector<float> gpu_rt(output_len" in cuda_rt_scale_bench
    assert cuda_rt_scale_bench.index("if (score_only)") < cuda_rt_scale_bench.index("std::vector<float> gpu_rt(output_len")

    for name, header_text in (("cuda", cuda_header), ("rocm", rocm_header)):
        assert "namespace kernel" in header_text, name
        assert "launch_continuous_chunk" in header_text, name
        assert "launch_mutual_info_chunk" in header_text, name
        assert "launch_spearman_chunk" in header_text, name
    assert "launch_selected_metric_max" in cuda_header
    assert "launch_accumulate_exceedances" in cuda_header
    assert "launch_decision_path_membership" not in cuda_header
    assert "decision_path_membership_kernel" in cuda_rt_header
    assert "GafimeRtBox" in cuda_rt_header
    assert "GafimeRtTriVertex" in cuda_rt_header
    assert "GafimeRtTriIndex" in cuda_rt_header
    assert "pack_decision_path_points_kernel" in cuda_rt_header
    assert "decision_path_bitset_kernel" in cuda_rt_header
    assert "score_decision_path_bitset_kernel" in cuda_rt_header
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_header
    assert "decision_path_target_stats_kernel" in cuda_rt_header
    assert "launch_decision_path_membership" in cuda_rt_launcher_header
    assert "execute_decision_path_membership" in cuda_rt_launcher_header
    assert "execute_decision_path_score" in cuda_rt_launcher_header

    assert "kernel void gafime_score_continuous" in metal_shader
    assert "kernel void gafime_score_mutual_info" in metal_shader
    assert "kernel void gafime_score_spearman" in metal_shader
    assert "placeholder" not in metal_shader.lower()
    assert "row * cols + combo" not in metal_shader
    assert "row * cols + col" not in metal_shader
    assert "row * info.cols" not in metal_shader
    assert "interaction_value(features, column_means, i, info.cols" not in metal_shader
    assert "interaction_value(features, column_means, j, info.cols" not in metal_shader
    assert "static_cast<ulong>(col) * rows + row" in metal_shader
    assert "interaction_value(features, column_means, row, info.rows" in metal_shader
    # Metal launcher exposes the same metric surface as CUDA/ROCm.
    assert "GAFIME_METRIC_MUTUAL_INFO" in metal_launcher and "GAFIME_METRIC_SPEARMAN" in metal_launcher
    assert "launcher.mm" in metal_cmake and "shader.metal" in metal_cmake

    assert "build_feature_major_host" in cuda_launcher
    assert "resident_features.data()" in cuda_launcher
    assert "build_feature_major_host" in rocm_launcher
    assert "resident_features.data()" in rocm_launcher
    assert "build_feature_major" in metal_launcher
    assert "resident_features.data()" in metal_launcher

    for marker in (
        "GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY",
        "GAFIME_GPU_DEVICE_FLAG_INTEGRATED",
        "GAFIME_GPU_DEVICE_FLAG_DISCRETE",
        "GAFIME_GPU_DEVICE_FLAG_AMD_RDNA",
        "GAFIME_GPU_DEVICE_FLAG_AMD_CDNA",
        "GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY",
        "GAFIME_GPU_ARCH_NVIDIA_ADA",
        "GAFIME_GPU_ARCH_AMD_CDNA",
        "GAFIME_GPU_ARCH_APPLE",
        "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT",
    ):
        assert marker in common_header, marker

    assert "cuda_arch_class" in cuda_launcher
    assert "cuda_device_flags" in cuda_launcher
    assert "cudaDriverGetVersion" in cuda_launcher
    assert "cudaRuntimeGetVersion" in cuda_launcher
    assert "cudaFuncSetCacheConfig" in cuda_launcher
    assert "cudaFuncAttributePreferredSharedMemoryCarveout" in cuda_launcher
    assert "tune_rt_kernels_for_device" in cuda_launcher
    assert "tune_rt_kernels_for_device" in cuda_rt_launcher

    assert "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES" in optix_smoke
    assert "__intersection__gafime_dp_box" in optix_smoke
    assert "__anyhit__gafime_dp_mark" in optix_smoke
    assert "open_lo_mask" in optix_smoke
    assert "sm_decision_path_membership_kernel" in optix_smoke
    assert "GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP" in cuda_abi_smoke
    assert "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT" in cuda_abi_smoke
    assert "gafime_gpu_decision_path_score" in cuda_abi_smoke

    assert "gcnArchName" in rocm_launcher
    assert "rocm_arch_is_rdna" in rocm_launcher
    assert "rocm_arch_is_cdna" in rocm_launcher
    assert "rocm_use_managed_memory" in rocm_launcher
    assert "hipMallocManaged" in rocm_launcher
    assert "Radeon Graphics" in rocm_launcher

    assert "hasUnifiedMemory" in metal_launcher
    assert "MTLResourceStorageModeManaged" in metal_launcher
    assert "didModifyRange" in metal_launcher
    assert "synchronizeResource" in metal_launcher
    assert "GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY" in metal_launcher
    assert "metal_payload_compile" in contract_workflow
    assert "cmake -S src/metal" in contract_workflow
    assert "xcrun -sdk macosx metal" in contract_workflow

    # Performance/optimization flags (e.g. -O3) are permitted because they do not
    # change the reference numerical result. Only math-breaking flags that relax
    # IEEE semantics are forbidden without maintainer approval, because they break
    # the f64/Kahan-accumulator parity oracle. See the "Compiler Ownership" section
    # of docs/contract.md.
    math_breaking_flags = (
        "-ffast-math",
        "--use_fast_math",
        "-Ofast",
        "-funsafe-math-optimizations",
        "-fassociative-math",
        "-freciprocal-math",
        "-ffinite-math-only",
        "-fno-signed-zeros",
        "-ffp-contract=fast",
        "-ftz=true",
        "/fp:fast",
    )
    for cmake_text in (cuda_cmake, rocm_cmake, metal_cmake):
        assert "host.cpp" not in cmake_text
        assert "tune.cpp" not in cmake_text
        assert "device.cu" not in cmake_text
        assert "device.hip" not in cmake_text
        assert "device.metal" not in cmake_text
        for banned in math_breaking_flags:
            assert banned not in cmake_text, f"math-breaking flag not allowed: {banned}"

    assert "kernels.cu" in cuda_cmake and "launcher.cu" in cuda_cmake
    assert "rt_kernels.cu" in cuda_cmake and "rt_launcher.cu" in cuda_cmake
    assert "GAFIME_CUDA_ENABLE_OPTIX_RT" in cuda_cmake
    assert "--ptx" in cuda_cmake and "rt_kernels.cu" in cuda_cmake
    assert "gafime_rt_optix_ptx.hpp" in cuda_cmake
    assert "CUDA::cuda_driver" in cuda_cmake
    assert "kernels.hip" in rocm_cmake and "launcher.hip" in rocm_cmake


def check_native_abi_and_reduce_scale_structure() -> None:
    types_text = (ROOT / "crates" / "gafime-types" / "src" / "lib.rs").read_text()
    assert "include_str!(" in types_text
    assert "src/common/gafime_gpu_abi.hpp" in types_text
    assert "gpu_abi_header_and_rust_layouts_stay_in_lockstep" in types_text
    assert "offset_of!(GafimeLaunchProtocol, permutations)" in types_text
    assert "offset_of!(GafimeResultTable, backend_private)" in types_text
    assert "GafimePermutationSignificanceTable" in types_text
    assert "GafimeDecisionPathTerm" in types_text
    assert "GafimeDecisionPathBatch" in types_text
    assert "GafimeDecisionPathScoreBatch" in types_text
    assert "gafime_gpu_permutation_pvalues" in (
        ROOT / "src" / "common" / "gafime_gpu_abi.hpp"
    ).read_text()
    assert "gafime_gpu_decision_path_membership" in (
        ROOT / "src" / "common" / "gafime_gpu_abi.hpp"
    ).read_text()
    assert "gafime_gpu_decision_path_score" in (
        ROOT / "src" / "common" / "gafime_gpu_abi.hpp"
    ).read_text()
    assert "supports_decision_path_membership" in (
        ROOT / "crates" / "gafime-gpu-sys" / "src" / "lib.rs"
    ).read_text()
    assert "supports_decision_path_score" in (
        ROOT / "crates" / "gafime-gpu-sys" / "src" / "lib.rs"
    ).read_text()
    assert "cuda_decision_path_direct_score_groups_mixed_axes_when_rt_is_required" in (
        ROOT / "crates" / "gafime-gpu-sys" / "src" / "lib.rs"
    ).read_text()
    assert "cuda_decision_path_direct_score_recomputes_target_stats_with_cached_points" in (
        ROOT / "crates" / "gafime-gpu-sys" / "src" / "lib.rs"
    ).read_text()
    assert "cuda_decision_path_direct_score_refreshes_cached_scatter_map" in (
        ROOT / "crates" / "gafime-gpu-sys" / "src" / "lib.rs"
    ).read_text()

    reduce_text = (ROOT / "crates" / "gafime-orchestrator" / "src" / "reduce" / "mod.rs").read_text()
    assert "CompactResultTablePlan" in reduce_text
    assert "planned_rows.min(rank.top_k as u64)" in reduce_text
    assert "10_000_000" in reduce_text
    assert "result_plan_bounds_ten_million_candidate_metadata_without_records" in reduce_text

    schedule_text = (ROOT / "crates" / "gafime-orchestrator" / "src" / "schedule" / "mod.rs").read_text()
    continuous_text = (ROOT / "crates" / "gafime-orchestrator" / "src" / "continuous.rs").read_text()
    assert "ContinuousSchedule" in schedule_text
    assert "CompactResultTablePlan::for_plan" in schedule_text
    assert "ContinuousSchedule::for_plan(&plan)" in continuous_text
    assert "self.schedule.result_table().capacity()" in continuous_text

    cpu_text = (ROOT / "crates" / "gafime-cpu" / "src" / "lib.rs").read_text()
    assert "execute_ranked_continuous" in cpu_text
    assert "TopKSelector" in cpu_text
    assert "planned_rows.min(protocol.rank.top_k as u64)" in cpu_text
    assert "OwnedResultTable::new(2, 1, 2)" in cpu_text


def check_pyo3_compact_report_and_cuda_surface() -> None:
    py_text = (ROOT / "crates" / "gafime-py" / "src" / "lib.rs").read_text()
    assert "table: OwnedResultTable" in py_text
    assert "records: Vec<PyContinuousRecord>" not in py_text
    assert "impl From<ContinuousReport> for PyContinuousReport" in py_text
    assert "table: value.table" in py_text
    assert "GpuBackend::cuda_from_env" in py_text
    assert "\"auto\" => Ok(resolve_auto_backend(device_id))" in py_text
    assert "probe_gpu_candidate(GAFIME_BACKEND_CUDA" in py_text
    assert "GpuBackend::metal_from_env" in py_text
    assert "cpu_isa_rank(finite_dispatch_isa())" in py_text
    assert "\"cuda\" => Ok(GAFIME_BACKEND_CUDA)" in py_text
    assert "\"gpu\" => Err" in py_text
    assert "v1-cuda-cabi" in py_text

    cargo_text = (ROOT / "crates" / "gafime-py" / "Cargo.toml").read_text()
    assert "gafime-gpu-sys" in cargo_text


def check_report_scale_view() -> None:
    sys.path.insert(0, str(ROOT / "python"))
    fake = install_fake_boundary(length=10_000_000)
    import gafime

    cfg = gafime.EngineConfig(
        metric_names=("pearson", "r2"),
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=10_000_000),
        permutation_tests=0,
        num_repeats=1,
    )
    report = gafime.GafimeEngine(cfg).analyze([[1.0, 0.0], [2.0, 1.0]], [1.0, 2.0], ["a", "b"])
    assert fake.calls
    assert len(report.interactions) == 10_000_000
    top = report.interactions.top_k(3, metric_name="pearson")
    assert len(top) == 3
    assert [item.combo for item in top] == [(0,), (1,), (0,)]


def run_cargo(include_gpu: bool) -> None:
    env = os.environ.copy()
    if include_gpu:
        env["RUST_TEST_THREADS"] = "1"
        required = (
            "/tmp/libgafime_cuda_v1.so",
            "/tmp/libgafime_rocm_v1.so",
            "/tmp/cuda_v1_abi_smoke",
            "/tmp/rocm_v1_abi_smoke",
        )
        missing = [path for path in required if not Path(path).exists()]
        if missing:
            raise AssertionError(f"missing optional GPU payloads or smokes: {missing}")
        env.setdefault("GAFIME_CUDA_V1_LIB", "/tmp/libgafime_cuda_v1.so")
        subprocess.run(["/tmp/cuda_v1_abi_smoke"], cwd=ROOT, check=True, env=env)
        subprocess.run(["/tmp/rocm_v1_abi_smoke"], cwd=ROOT, check=True, env=env)
    subprocess.run(["cargo", "test", "--workspace"], cwd=ROOT, check=True, env=env)
    if include_gpu:
        subprocess.run(["cargo", "test", "-p", "gafime-gpu-sys"], cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--skip-cargo", action="store_true")
    args = parser.parse_args()

    check_no_local_legacy_runtime_artifacts()
    check_packaging()
    check_no_source_opt_in_or_fallback()
    check_native_kernel_structure()
    check_native_abi_and_reduce_scale_structure()
    check_pyo3_compact_report_and_cuda_surface()
    check_runtime_surface()
    check_report_scale_view()
    if not args.skip_cargo:
        run_cargo(args.include_gpu)
    print("v1 architecture gate passed")


if __name__ == "__main__":
    main()
