#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def read_rust_crate_sources(
    crate_name: str,
    *,
    include_test_modules: bool = True,
    include_local_cmake_experiment: bool = True,
) -> str:
    source_root = ROOT / "crates" / crate_name / "src"
    paths = sorted(source_root.rglob("*.rs"))
    if not include_test_modules:
        paths = [
            path for path in paths if "tests" not in path.relative_to(source_root).parts
        ]
    if not include_local_cmake_experiment:
        paths = [path for path in paths if "local_cmake_experiment" not in path.name]
    return "\n".join(path.read_text() for path in paths)


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
        return FakeRecord(
            self.combo(index), self.metric_values(index), self.candidate_id(index)
        )

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
        raise AssertionError(
            "v1 gate forbids normal-path Python report list materialization"
        )


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
    assert (
        pyproject["tool"]["maturin"]["manifest-path"] == "crates/gafime-py/Cargo.toml"
    )
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

    packaged_files = {
        path.relative_to(ROOT / "python").as_posix()
        for path in (ROOT / "python").rglob("*")
        if path.is_file()
    }
    forbidden_paths = {
        "gafime/backends",
        "gafime/native_data.py",
        "gafime/compile/sessions.py",
    }
    for forbidden in forbidden_paths:
        assert all(not item.startswith(forbidden) for item in packaged_files), forbidden

    package_text = "\n".join(
        path.read_text() for path in (ROOT / "python" / "gafime").rglob("*.py")
    )
    for forbidden in FORBIDDEN_RUNTIME_STRINGS:
        assert forbidden not in package_text, forbidden


def check_compatibility_scenario_metadata() -> None:
    source = (ROOT / "python/gafime/compile/scenario.py").read_text()
    for forbidden in (
        "itertools",
        "numpy",
        "gafime_py",
        "compile_continuous",
        "execute_continuous",
        "candidate_indices",
        "for row in",
        "for combo in",
    ):
        assert forbidden not in source, forbidden

    from gafime import CompileFlags, ComputeBudget, EngineConfig
    from gafime.compile.scenario import build_scenario_plan_from_shape

    plan = build_scenario_plan_from_shape(
        n_samples=1_000_000_000,
        n_features=1_000_000,
        config=EngineConfig(
            budget=ComputeBudget(
                max_comb_size=5,
                max_combinations_per_k=100,
                top_features_for_higher_k=50,
            )
        ),
        flags=CompileFlags(plan=True),
    )
    assert plan.n_samples == 1_000_000_000
    assert plan.n_features == 1_000_000
    assert len(plan.continuous) == 5
    assert plan.planned_count == 500
    assert all(item.planned_count == 100 for item in plan.continuous)
    assert all(not hasattr(item, "candidate_indices") for item in plan.continuous)


def check_no_local_legacy_runtime_artifacts() -> None:
    offenders = []
    for relative in FORBIDDEN_LOCAL_RUNTIME_PATHS:
        path = ROOT / relative
        if path.exists():
            offenders.append(relative)
    for pattern in FORBIDDEN_LOCAL_RUNTIME_GLOBS:
        offenders.extend(
            path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern)
        )
    assert not offenders, (
        f"legacy local runtime artifacts are present: {sorted(offenders)}"
    )


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
    loaded_forbidden = FORBIDDEN_RUNTIME_MODULES.intersection(
        set(sys.modules) - before_modules
    )
    assert not loaded_forbidden, sorted(loaded_forbidden)


def check_no_source_opt_in_or_fallback() -> None:
    source_text = "\n".join(
        path.read_text() for path in (ROOT / "python" / "gafime").rglob("*.py")
    )
    assert "GAFIME_V1_ENGINE" not in source_text
    assert "GAFIME_USE_LEGACY_ENGINE" not in source_text


def check_native_kernel_structure() -> None:
    simd_root = ROOT / "crates" / "gafime-cpu" / "src" / "simd"
    simd_mod_text = (simd_root / "mod.rs").read_text()
    isa_text = (simd_root / "isa.rs").read_text()
    covariance_text = (simd_root / "covariance.rs").read_text()
    covariance_f32_text = (simd_root / "covariance_f32.rs").read_text()
    histogram_text = (simd_root / "histogram.rs").read_text()
    dispatch_text = (ROOT / "crates" / "gafime-cpu" / "src" / "dispatch.rs").read_text()
    assert "pub use crate::simd::*" in dispatch_text
    assert "mod covariance" in simd_mod_text
    assert "mod covariance_f32" in simd_mod_text
    assert "mod histogram" in simd_mod_text
    assert "mod isa" in simd_mod_text
    assert "finite_dispatch_isa" in isa_text
    assert "pearson_sums_avx2" in covariance_text
    assert "pearson_sums_sse42" in covariance_text
    assert "pearson_sums_neon" in covariance_text
    assert "fixed_bin_histogram2d" in histogram_text
    assert "fixed_bin_histogram2d_avx2" in histogram_text
    assert "pub fn fixed_bin_indices_into" in histogram_text
    assert "fixed_bin_indices_into" in simd_mod_text
    assert "fixed_bins_from_scaled_avx2" in histogram_text
    assert "_mm256_cvttps_epi32" in histogram_text
    assert "_mm256_blendv_epi8" in histogram_text
    assert "scaled_lanes" not in histogram_text
    assert '#[target_feature(enable = "avx2")]' in covariance_text
    assert '#[target_feature(enable = "sse4.2")]' in covariance_text
    assert '#[target_feature(enable = "avx2")]' in histogram_text
    dispatch_body = covariance_text.split("fn pearson_sums_dispatched", 1)[1].split(
        "unsafe fn pearson_sums_avx512",
        1,
    )[0]
    assert "pearson_sums_avx2" in dispatch_body
    assert "pearson_sums_sse42" in dispatch_body
    assert "pearson_sums_neon" in dispatch_body
    assert "all_pairs_finite" not in covariance_text
    assert "EARLY_NONFINITE_PROBE_ROWS: usize = 16" in covariance_text
    assert "all_finite_avx512_pd" in covariance_text
    assert "all_finite_avx2_pd" in covariance_text
    assert "all_finite_sse_pd" in covariance_text
    assert "all_finite_neon_f64" in covariance_text

    matrix_text = (ROOT / "crates" / "gafime-cpu" / "src" / "matrix.rs").read_text()
    assert "columns: Vec<f32>" in matrix_text
    assert "transpose_row_major" in matrix_text
    assert "self.columns[col * self.rows as usize + row]" in matrix_text

    kernels_text = (
        ROOT / "crates" / "gafime-cpu" / "src" / "kernels" / "mod.rs"
    ).read_text()
    cpu_precision_kernels = (
        ROOT / "crates" / "gafime-cpu" / "src" / "kernels" / "precision.rs"
    ).read_text()
    assert "cached_pearson" in kernels_text
    assert "get_or_insert_with(|| pearson(signal, matrix.target()))" in kernels_text
    assert "ContinuousScoreScratch" in kernels_text
    assert "score_continuous_combo_into" in kernels_text
    assert "matrix.column(combo[0] as usize)" in kernels_text
    assert "simd::fixed_bin_histogram2d" in kernels_text
    assert "simd::fixed_bin_indices(&x_values" not in kernels_text
    assert "Vec::with_capacity(rows)" not in kernels_text
    histogram_avx2 = histogram_text.split("unsafe fn fixed_bin_histogram2d_avx2", 1)[
        1
    ].split("#[cfg(test)]", 1)[0]
    assert "hist_x[x_bin] += 1" in histogram_avx2
    assert "hist_y[y_bin] += 1" in histogram_avx2
    assert "joint[x_bin * bins_usize + y_bin] += 1" in histogram_avx2
    assert "get_unchecked" not in histogram_avx2

    native_root = ROOT / "src"
    rust_gpu_crate = ROOT / "crates" / "gafime-gpu-sys" / "src"
    cuda_root = native_root / "cuda"
    rocm_root = native_root / "rocm"
    metal_root = native_root / "metal"
    common_root = native_root / "common"

    assert not (ROOT / "gpu").exists(), (
        "v1 GPU runtime sources must live under root src/"
    )
    for stale_root in ("common", "cuda", "rocm", "metal"):
        stale_path = rust_gpu_crate / stale_root
        if stale_path.exists():
            assert not any(path.is_file() for path in stale_path.rglob("*")), (
                "crates/gafime-gpu-sys/src must stay Rust-only; native sources belong under root src/"
            )
    assert (common_root / "gafime_gpu_abi.hpp").exists()
    assert (common_root / "gafime_gpu_exports.map").exists()
    assert (common_root / "gafime_gpu_internal_abi.hpp").exists()
    assert (common_root / "gpu_abi_impl.hpp").exists()
    assert (cuda_root / "cuda_api.hpp").exists()
    assert (cuda_root / "cuda_internal.hpp").exists()
    assert (cuda_root / "rt_abi.hpp").exists()
    assert (rocm_root / "rocm_api.hpp").exists()
    assert (metal_root / "metal_api.hpp").exists()

    for backend_root in (cuda_root, rocm_root, metal_root, common_root):
        for path in backend_root.iterdir():
            if path.name == "CMakeLists.txt":
                continue
            if path.name == "embed_ptx.cmake":
                assert backend_root == cuda_root
                continue
            if path.name == "precision_abi_smoke.cpp":
                assert backend_root in {cuda_root, rocm_root}
                cmake = (backend_root / "CMakeLists.txt").read_text()
                assert "BUILD_TESTS" in cmake
                assert '"${CMAKE_CURRENT_LIST_DIR}/precision_abi_smoke.cpp"' in cmake
                continue
            assert path.suffix in {
                ".hpp",
                ".cuh",
                ".cu",
                ".hip",
                ".map",
                ".metal",
                ".mm",
            }, path
            assert path.suffix not in {".h", ".cpp"}, path

    cuda_api = (cuda_root / "cuda_api.hpp").read_text()
    cuda_internal = (cuda_root / "cuda_internal.hpp").read_text()
    cuda_rt_abi = (cuda_root / "rt_abi.hpp").read_text()
    cuda_header = (cuda_root / "kernels.cuh").read_text()
    cuda_precision_kernels = (cuda_root / "precision_kernels.cu").read_text()
    cuda_precision_header = (cuda_root / "precision_kernels.cuh").read_text()
    cuda_precision_launcher = (cuda_root / "precision_launcher.cu").read_text()
    cuda_kernels = cuda_precision_kernels
    cuda_launcher = cuda_precision_launcher
    cuda_rt_launcher = (cuda_root / "rt_launcher.cu").read_text()
    cuda_rt_kernels = (cuda_root / "rt_kernels.cu").read_text()
    cuda_rt_header = (cuda_root / "rt_kernels.cuh").read_text()
    cuda_rt_launcher_header = (cuda_root / "rt_launcher.cuh").read_text()
    cuda_cmake = (cuda_root / "CMakeLists.txt").read_text()
    cuda_rt_exports_map = (cuda_root / "gafime_gpu_rt_exports.map").read_text()
    rocm_launcher = (rocm_root / "launcher.hip").read_text()
    rocm_kernels = (rocm_root / "kernels.hip").read_text()
    rocm_header = (rocm_root / "kernels.hpp").read_text()
    rocm_precision = (rocm_root / "precision.hpp").read_text()
    rocm_cmake = (rocm_root / "CMakeLists.txt").read_text()
    rocm_native_doc = (ROOT / "docs" / "rocm-native-timing.md").read_text()
    metal_launcher = (metal_root / "launcher.mm").read_text()
    metal_shader = (metal_root / "shader.metal").read_text()
    metal_cmake = (metal_root / "CMakeLists.txt").read_text()
    stage_gpu_payload = (
        ROOT / ".github" / "scripts" / "stage_gpu_payload.py"
    ).read_text()
    common_header = (common_root / "gafime_gpu_abi.hpp").read_text()
    common_exports_map = (common_root / "gafime_gpu_exports.map").read_text()
    python_config = (ROOT / "python" / "gafime" / "config.py").read_text()
    python_precision = (ROOT / "python" / "gafime" / "_precision.py").read_text()
    python_capabilities = (ROOT / "python" / "gafime" / "capabilities.py").read_text()
    python_adapter = (ROOT / "python" / "gafime" / "v1_adapter.py").read_text()
    rust_runtime = (ROOT / "crates" / "gafime-py" / "src" / "runtime.rs").read_text()
    rust_precision_ranking = (
        ROOT / "crates" / "gafime-py" / "src" / "continuous.rs"
    ).read_text()
    rust_public_precision = "\n".join(
        (ROOT / "crates" / "gafime-py" / "src" / name).read_text()
        for name in ("common.rs", "continuous.rs", "generated.rs", "py_api.rs")
    )
    python_decision_path = (ROOT / "python" / "gafime" / "decision_path.py").read_text()
    precision_doc = (ROOT / "docs" / "precision-contract.md").read_text()
    interaction_diagnostic_evidence = (
        ROOT / "docs" / "evidence" / "interaction-overflow-diagnostics.md"
    ).read_text()
    cpu_covariance_evidence = (
        ROOT / "docs" / "evidence" / "cpu-covariance-finite-pass.md"
    ).read_text()
    cpu_mi_evidence = (
        ROOT / "docs" / "evidence" / "cpu-mi-branchless-bins.md"
    ).read_text()
    cpu_mi_perf = (
        ROOT / "tests" / "release_measure" / "perf_11_cpu_mi_histogram.py"
    ).read_text()
    continuous_combos = (
        ROOT / "crates" / "gafime-orchestrator" / "src" / "plan" / "combos.rs"
    ).read_text()
    static_kernel_report = (
        ROOT / "tests" / "release_measure" / "gpu_static_kernel_report.py"
    ).read_text()
    contract_workflow = (
        ROOT / ".github" / "workflows" / "v1_contract_validation.yml"
    ).read_text()
    native_validation_workflow = (
        ROOT / ".github" / "workflows" / "native_platform_validation.yml"
    ).read_text()
    metal_lab_workflow = (
        ROOT / ".github" / "workflows" / "metal_macos26_lab.yml"
    ).read_text()
    metal_beast_workflow = (
        ROOT / ".github" / "workflows" / "metal_beast_benchmark.yml"
    ).read_text()
    metal_native_timing = (
        ROOT / "tests" / "gpu" / "metal_precision_native_timing.mm"
    ).read_text()
    cuda_native_timing = (
        ROOT / "tests" / "gpu" / "cuda_precision_native_timing.cu"
    ).read_text()
    rocm_native_timing = (
        ROOT / "tests" / "gpu" / "rocm_native_timing.cpp"
    ).read_text()
    core_native_timing = (
        ROOT
        / "crates"
        / "gafime-cpu"
        / "tests"
        / "precision_native_benchmark.rs"
    ).read_text()
    core_native_standalone = (
        ROOT
        / "tests"
        / "release_measure"
        / "core_precision_native_benchmark.rs"
    ).read_text()
    core_native_runner = (
        ROOT
        / "tests"
        / "release_measure"
        / "run_core_precision_native_benchmark.py"
    ).read_text()
    precision_profile_perf = (
        ROOT / "tests" / "release_measure" / "perf_13_precision_profiles.py"
    ).read_text()
    canonical_lifecycle_evidence = (
        ROOT
        / "tests"
        / "release_measure"
        / "canonical_abi_lifecycle_evidence.py"
    ).read_text()
    cold_lifecycle = (
        ROOT / "tests" / "release_measure" / "cold_lifecycle.py"
    ).read_text()
    cuda_abi_smoke = (ROOT / "tests" / "gpu" / "cuda_v1_abi_smoke.cpp").read_text()
    rocm_abi_smoke = (ROOT / "tests" / "gpu" / "rocm_v1_abi_smoke.cpp").read_text()
    optix_smoke = (
        ROOT / "tests" / "gpu" / "cuda_rt_decision_path_optix_smoke.cu"
    ).read_text()
    cuda_rt_scale_bench = (
        ROOT / "tests" / "gpu" / "cuda_rt_membership_scale_bench.cpp"
    ).read_text()
    cuda_rt_state_policy = (
        ROOT / "tests" / "gpu" / "cuda_rt_state_policy_test.cpp"
    ).read_text()
    cuda_rt_concurrency = (
        ROOT / "tests" / "gpu" / "cuda_rt_same_device_concurrency.cpp"
    ).read_text()
    cuda_rt_paper = (
        ROOT / "docs" / "rt-gbdt-hardware-ray-tracing-paper.tex"
    ).read_text()
    cuda_rt_paper_repro = (ROOT / "docs" / "rt-gbdt-paper-repro.md").read_text()

    for fixture, protocol_names in (
        (
            cuda_abi_smoke,
            ("protocol", "stable_protocol", "mi_protocol", "mixed_protocol"),
        ),
        (rocm_abi_smoke, ("protocol", "stable_protocol")),
    ):
        for protocol_name in protocol_names:
            assert f"{protocol_name}.family_count = 1;" in fixture, (
                f"standalone ABI fixture leaves {protocol_name}.family_count unset"
            )
    assert "mixed_chunks[1].combo_row_offset = 3;" in cuda_abi_smoke
    assert "mixed_chunks[1].local_chunk_id = 1;" in cuda_abi_smoke
    assert "chunks[1].combo_row_offset = 3;" in rocm_abi_smoke
    assert "chunks[1].local_chunk_id = 1;" in rocm_abi_smoke

    assert "GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL 0x4u" in common_header
    assert "GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL 0x200u" in common_header
    assert "GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION 0x400u" in common_header
    assert "GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64 0x800u" in common_header
    assert "GAFIME_GPU_DEVICE_FLAG_F64_STORAGE 0x1000u" in common_header
    assert "GAFIME_DTYPE_F64 = 2" in common_header
    assert "GAFIME_NUMERIC_ROUTE_ABI_MIN_MINOR 1u" in common_header
    assert "GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT 0u" in common_header
    assert "gafime.abi-1.1-consumer-result.v1" in canonical_lifecycle_evidence
    assert "independent_abi_1_1_c_consumer" in canonical_lifecycle_evidence
    assert "payload_bytes != embedded_payload" in canonical_lifecycle_evidence
    assert "canonical lifecycle evidence requires a clean source tree" in (
        canonical_lifecycle_evidence
    )
    for canonical_operation in (
        "numeric_routes",
        "matrix_alloc",
        "matrix_upload",
        "matrix_update_target",
        "execute",
        "execution_memory_peak",
        "permutation_memory_peak",
        "permutation_pvalues",
        "interaction_diagnostics",
        "matrix_free",
    ):
        assert f'"{canonical_operation}"' in canonical_lifecycle_evidence
    assert "canonical_payload_lifecycle_independent_consumer_required" in (
        precision_profile_perf
    )
    for structure in (
        "GafimeNumericRoute",
        "GafimeConstBufferView",
        "GafimeMutableBufferView",
        "GafimeNumericMatrixDesc",
        "GafimeNumericLaunchProtocol",
        "GafimeNumericResultTable",
        "GafimeNumericSignificanceTable",
    ):
        assert structure in common_header
    canonical_numeric_symbols = (
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_matrix_update_target_v2",
        "gafime_gpu_execute_v2",
        "gafime_gpu_execution_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2",
        "gafime_gpu_matrix_free_v2",
    )
    for symbol in canonical_numeric_symbols:
        assert symbol in common_header
        for launcher_text in (cuda_launcher, rocm_launcher, metal_launcher):
            assert symbol in launcher_text
    for suffix_symbol in (
        "gafime_gpu_matrix_upload_f32_v2",
        "gafime_gpu_matrix_upload_f64_v2",
        "gafime_gpu_matrix_update_target_f32_v2",
        "gafime_gpu_matrix_update_target_f64_v2",
        "gafime_gpu_execute_f32_v2",
        "gafime_gpu_execute_f64_v2",
        "gafime_gpu_permutation_pvalues_f32_v2",
        "gafime_gpu_permutation_pvalues_f64_v2",
    ):
        assert suffix_symbol not in common_header
        for launcher_text in (cuda_launcher, rocm_launcher, metal_launcher):
            assert suffix_symbol not in launcher_text
    for name, launcher_text in (
        ("cuda", cuda_launcher),
        ("rocm", rocm_launcher),
        ("metal", metal_launcher),
    ):
        assert "GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL" in launcher_text, name
        assert "GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL" in launcher_text, name
        assert "GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION" in launcher_text, name
        assert "descriptors_resident" in launcher_text, name
        assert (
            "invalidate_precision_descriptors" in launcher_text
            if name == "cuda"
            else (
                "precision_invalidate_descriptor_cache" in launcher_text
                if name == "rocm"
                else "invalidate_protocol_descriptor_cache" in launcher_text
            )
        ), name
        assert "GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT" in launcher_text, (
            name
        )
        assert (
            "base->reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] != 0"
            in launcher_text
            if name == "cuda"
            else (
                "descriptor_generation == generation" in launcher_text
                if name == "rocm"
                else "descriptor_generation != 0" in launcher_text
            )
        ), name
        assert "descriptor_combo_host_ptr" not in launcher_text, name
        assert "descriptor_metric_ids_host_ptr" not in launcher_text, name
    for launcher_text in (cuda_launcher, rocm_launcher):
        assert "GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64" in launcher_text
    for launcher_text in (cuda_launcher, rocm_launcher, metal_launcher):
        assert "flags |= GAFIME_GPU_DEVICE_FLAG_F64_STORAGE" not in launcher_text
    for fixture in (cuda_abi_smoke, rocm_abi_smoke):
        assert "GAFIME_EXPECT_MI_ACCUMULATION_FP64" in fixture
        assert "GAFIME_DTYPE_F64" in fixture
        assert "did not fail closed on f64 storage" in fixture
    assert "HipDeviceBufferReservation" in rocm_launcher
    cuda_prepare = cuda_launcher.split("int prepare_precision_buffers(", 1)[1].split(
        "uint32_t primary_metric_index(", 1
    )[0]
    cache_invalidation = cuda_prepare.index("invalidate_precision_descriptors(matrix)")
    first_reservation = cuda_prepare.index("&matrix->combo_indices", cache_invalidation)
    second_reservation = cuda_prepare.index("&matrix->metric_ids", first_reservation)
    first_copy = cuda_prepare.index("cudaMemcpy(", second_reservation)
    assert cache_invalidation < first_reservation < second_reservation < first_copy

    assert "struct DescriptorBufferUpdateTransition" in rocm_launcher
    assert "DescriptorBufferUpdateTransition{true, false, false, false}" in rocm_launcher
    assert "DescriptorBufferUpdateTransition{true, true, true, false}" in rocm_launcher
    first_reservation = rocm_launcher.index(
        "HipDeviceBufferReservation<uint32_t> combo_reservation"
    )
    second_reservation = rocm_launcher.index(
        "HipDeviceBufferReservation<uint32_t> metric_reservation",
        first_reservation,
    )
    cache_invalidation = rocm_launcher.index(
        "precision_invalidate_descriptor_cache(matrix)", second_reservation
    )
    assert first_reservation < second_reservation < cache_invalidation
    assert "replaces_live_buffer()" in rocm_launcher[second_reservation:]
    for fixture in (cuda_abi_smoke, rocm_abi_smoke):
        assert "descriptor_generation_reused_address" in fixture
        assert "descriptor_generation_replay" in fixture
        assert "descriptor_generation_legacy_repeat" in fixture
        assert (
            "protocol.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 102;"
            in fixture
        )
    continuous_execution = (
        ROOT / "crates" / "gafime-orchestrator" / "src" / "continuous.rs"
    ).read_text()
    assert (
        "protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT]" in continuous_execution
    )
    assert "descriptor_generation: next_descriptor_generation()" in continuous_execution
    gpu_sys = read_rust_crate_sources(
        "gafime-gpu-sys",
        include_test_modules=False,
        include_local_cmake_experiment=False,
    )
    gpu_sys_local = read_rust_crate_sources(
        "gafime-gpu-sys", include_test_modules=False
    )
    gpu_sys_with_tests = read_rust_crate_sources("gafime-gpu-sys")
    assert "supports_immutable_protocol" in gpu_sys
    assert "supports_descriptor_generation" in gpu_sys
    assert (
        "descriptor_generation_is_sent_only_to_generation_capable_payloads"
        in gpu_sys_with_tests
    )
    assert (
        "legacy_cuda_decision_path_payloads_share_a_host_execution_lock"
        in gpu_sys_with_tests
    )
    assert "legacy_cuda_decision_path_lock" not in gpu_sys
    assert "acquire_local_cmake_experiment_lock" in gpu_sys_local
    assert "negotiated.flags &= !GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL" in gpu_sys
    assert (
        "negotiated.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0"
        in gpu_sys
    )
    assert "cargo +1.89.0 test --workspace --locked --quiet" in contract_workflow
    assert "cargo +1.97.1 test --workspace --locked --quiet" in contract_workflow
    assert (
        "cargo +1.97.1 clippy --workspace --all-targets --locked -- -D warnings"
        in contract_workflow
    )
    assert '"$GITHUB_WORKSPACE/tests/python"' in contract_workflow
    assert (
        "metal_descriptor_cache_generation_refreshes_reused_addresses_when_available"
        in contract_workflow
    )
    assert "PRE_GENERATION_METAL_PAYLOAD_COMMIT" in native_validation_workflow
    assert "3f4cd842e08c7b6796bd03d0d63b9c100fc7b478" in native_validation_workflow
    assert "GAFIME_CUDA_BUILD_TESTS=ON" in contract_workflow
    assert "cuda_rt_state_policy_test.cpp" in cuda_cmake
    assert "cuda_rt_same_device_concurrency.cpp" in cuda_cmake
    assert "SKIP_RETURN_CODE 77" in cuda_cmake
    assert "gafime_gpu_decision_path_release_device_state" not in common_header
    assert "gafime_gpu_decision_path_release_device_state" in cuda_rt_abi
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" in cuda_abi_smoke
    assert "disabled_firsthit_status" in cuda_abi_smoke
    assert "cudaSetDevice(1) before RT calls" in cuda_rt_concurrency
    assert "cudaGetDevice after RT cleanup" in cuda_rt_concurrency
    assert "gafime_gpu_decision_path_release_device_state" in cuda_rt_state_policy
    assert "gafime_gpu_rt_exports.map" in cuda_cmake
    assert "gafime_gpu_*;" not in common_exports_map
    assert "gafime_gpu_*;" not in cuda_rt_exports_map
    for symbol in (
        "gafime_gpu_device_info",
        "gafime_gpu_graph_capability",
        "gafime_gpu_matrix_alloc",
        "gafime_gpu_matrix_upload",
        "gafime_gpu_matrix_update_target",
        "gafime_gpu_matrix_free",
        "gafime_gpu_execute",
        "gafime_gpu_execution_memory_peak",
        "gafime_gpu_permutation_memory_peak",
        "gafime_gpu_permutation_pvalues",
        "gafime_gpu_interaction_diagnostics",
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_matrix_alloc_v2",
        "gafime_gpu_matrix_upload_v2",
        "gafime_gpu_matrix_update_target_v2",
        "gafime_gpu_execute_v2",
        "gafime_gpu_execution_memory_peak_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2",
        "gafime_gpu_matrix_free_v2",
    ):
        assert f"{symbol};" in common_exports_map
        assert f"{symbol};" in cuda_rt_exports_map
    for symbol in (
        "gafime_gpu_decision_path_membership",
        "gafime_gpu_decision_path_score",
        "gafime_gpu_decision_path_release_device_state",
    ):
        assert f"{symbol};" in cuda_rt_exports_map
        assert f"{symbol};" not in common_exports_map
        assert symbol in cuda_rt_abi
        assert symbol not in cuda_precision_launcher
        assert symbol not in rocm_launcher
        assert symbol not in metal_launcher

    for name, launcher_text in (
        ("cuda", cuda_launcher),
        ("rocm", rocm_launcher),
        ("metal", metal_launcher),
    ):
        assert "__global__" not in launcher_text, f"{name} launcher owns device kernels"
        assert "__device__" not in launcher_text, f"{name} launcher owns device helpers"
        assert "placeholder" not in launcher_text.lower(), name
    assert "__global__" not in cuda_rt_launcher
    assert "__device__" not in cuda_rt_launcher
    assert "placeholder" not in cuda_rt_launcher.lower()

    assert "<<<" not in cuda_launcher, "CUDA ABI launcher dispatches through one kernel table"
    assert "<<<" in cuda_rt_launcher, "CUDA RT launcher owns RT <<<>>> launch calls"
    assert "<<<" in cuda_kernels, "CUDA typed kernel wrappers own specialized launches"
    assert "cuda_precision_kernel_set" in cuda_launcher
    assert "<<<" not in cuda_rt_kernels, "CUDA RT kernels file must not own launches"
    assert "hipLaunchKernelGGL" in rocm_launcher, "ROCm launcher owns HIP launch calls"
    assert "hipLaunchKernelGGL" not in rocm_kernels and "<<<" not in rocm_kernels

    for name, device_text in (("cuda", cuda_kernels), ("rocm", rocm_kernels)):
        continuous_marker = (
            "__global__ void continuous_kernel"
            if name == "cuda"
            else "__global__ void score_continuous_chunk_kernel"
        )
        spearman_marker = (
            "__global__ void spearman_kernel"
            if name == "cuda"
            else "__global__ void score_spearman_chunk_kernel"
        )
        mi_marker = (
            "__global__ void mutual_info_kernel"
            if name == "cuda"
            else "__global__ void score_mutual_info_chunk_kernel"
        )
        assert continuous_marker in device_text, name
        assert "__global__ void target_stats_kernel" in device_text, name
        assert "__global__ void unary_feature_stats_kernel" in device_text, name
        if name == "rocm":
            assert (
                "__global__ void score_continuous_unary_all_finite_chunk_kernel"
                in device_text
            ), name
        assert "target_stats->finite" in device_text, name
        assert "feature_stats[column].sxx" in device_text, name
        assert spearman_marker in device_text, name
        assert mi_marker in device_text, name
        assert "placeholder" not in device_text.lower(), name
        assert "row * cols + combo" not in device_text, (
            f"{name} kernels must not scan sample-major features"
        )
        assert "row * cols + col" not in device_text, (
            f"{name} kernels must not scan sample-major features"
        )
        assert "row * n_features" not in device_text, (
            f"{name} kernels must not scan sample-major features"
        )
        assert (
            "interaction_value(features, column_means, i, n_features" not in device_text
        ), f"{name} kernels must not pass feature-count as the feature-major stride"
        assert (
            "interaction_value(features, column_means, j, n_features" not in device_text
        ), f"{name} kernels must not pass feature-count as the feature-major stride"
        assert (
            "static_cast<uint64_t>(column) * rows + row" in device_text
            if name == "cuda"
            else "static_cast<uint64_t>(col) * rows + row" in device_text
        ), (
            f"{name} kernels must read feature-major resident features"
        )
        row_stride_marker = (
            "features, column_means, row, n_samples, combo"
            if name == "cuda"
            else "features, column_means, row, n_samples, combo"
        )
        assert row_stride_marker in device_text, (
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
    assert "rt_canonical_float_bits" in cuda_rt_header
    assert "rt_ordered_float_key" in cuda_rt_header
    assert "rt_float_bucket" in cuda_rt_header
    assert "kRtFloatBucketShift = 9u" in cuda_rt_header
    assert "optixGetAttribute_0" in cuda_rt_kernels
    assert "inside_box(point, path_idx)" in cuda_rt_kernels
    assert "GafimeRtTriVertex" in cuda_rt_header
    assert "GafimeRtTriIndex" in cuda_rt_header
    assert "pack_decision_path_points_kernel" in cuda_rt_kernels
    assert "pack_grouped_decision_path_points_kernel" in cuda_rt_kernels
    assert "scatter_decision_path_score_metrics_kernel" in cuda_rt_kernels
    assert "rt_kernel::decision_path_membership_kernel" in cuda_rt_launcher
    assert "optixLaunch" in cuda_rt_launcher
    assert "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES" in cuda_rt_launcher
    assert "OPTIX_BUILD_INPUT_TYPE_TRIANGLES" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_GEOMETRY" not in cuda_rt_launcher
    assert "RtGeometryMode::Triangle2dInstanced" in cuda_rt_launcher
    assert "rt_box_plan_triangle2d_is_safe" in cuda_rt_launcher
    assert "expand_rt_triangle_bound" in cuda_rt_launcher
    assert "span >= std::ldexp(scale, -12)" in cuda_rt_launcher
    assert "step < 8u" in cuda_rt_launcher
    assert "triangle_2d_instanced && !inside_box" in cuda_rt_kernels
    assert "make_rt_conservative_aabb" in cuda_rt_launcher
    assert "rt_float_bucket" in cuda_rt_launcher
    assert "kRtFloatEncodingVersion" in cuda_rt_launcher
    assert "OPTIX_DEVICE_PROPERTY_RTCORE_VERSION" in cuda_rt_launcher
    assert "rt_plan_signature" in cuda_rt_launcher
    assert "execute_decision_path_score" in cuda_rt_launcher
    assert "score_decision_path_bitset_kernel" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" in cuda_rt_launcher
    assert "build_rt_score_groups" in cuda_rt_launcher
    assert "prefer_direct_pair_groups" in cuda_rt_launcher
    assert "rt_score_first_hit_direct_requested" in cuda_rt_launcher
    assert "rt_box_plan_non_overlapping_2d" in cuda_rt_launcher
    assert "all_groups_non_overlapping_2d" in cuda_rt_launcher
    assert "decision_path_score_needs_duplicate_guard" in cuda_rt_launcher_header
    assert "decision_path_score_needs_duplicate_guard" in cuda_rt_launcher
    assert (
        "needs_duplicate_guard ? program.membership_words_device : nullptr"
        in cuda_rt_launcher
    )
    assert (
        "direct_first_hit && !grouped_plan.all_groups_non_overlapping_2d"
        in cuda_rt_launcher
    )
    assert "group.axes == path_axes" in cuda_rt_launcher
    assert "execute_decision_path_score_optix_grouped" in cuda_rt_launcher
    assert "execute_decision_path_score_optix_grouped_instanced" in cuda_rt_launcher
    assert "RtGeometryMode::CustomAabbInstanced" in cuda_rt_launcher
    assert (
        "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING" in cuda_rt_launcher
    )
    assert "OPTIX_BUILD_INPUT_TYPE_INSTANCES" in cuda_rt_launcher
    assert "rt_instanced_group_signature" in cuda_rt_launcher
    assert "bool rebuild_geometry" in cuda_rt_launcher
    assert "program.gas_signature = geometry_signature" in cuda_rt_launcher
    assert "params.handle = program.gas_handle" in cuda_rt_launcher
    assert "point_group_stride" in cuda_rt_launcher
    assert "grouped_point_stride = 3u" in cuda_rt_launcher
    assert "params.point_stride = grouped_point_stride" in cuda_rt_launcher
    assert "row * point_stride" in cuda_rt_kernels
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
    grouped_body = cuda_rt_launcher.split(
        "int execute_decision_path_score_optix_grouped(\n", 1
    )[1]
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
    planned_body = cuda_rt_launcher.split(
        "int execute_decision_path_score_optix_planned(\n", 1
    )[1]
    planned_body = planned_body.split(
        "int execute_decision_path_score_optix_grouped_instanced(\n", 1
    )[0]
    assert "decision_path_score_needs_duplicate_guard" in planned_body
    assert "status == GAFIME_STATUS_OK && needs_duplicate_guard" in planned_body
    assert (
        "needs_duplicate_guard ? program.membership_words_device : nullptr"
        in planned_body
    )
    assert (
        "params.membership_words = program.membership_words_device;" not in planned_body
    )
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_launcher
    assert "decision_path_target_stats_kernel" in cuda_rt_launcher
    assert "direct_inside_counts_device" in cuda_rt_launcher
    assert "direct_inside_sum_y_device" in cuda_rt_launcher
    assert "score_decision_path_direct_stats_scatter_kernel" in cuda_rt_launcher
    assert "write_decision_path_score_metadata_host" in cuda_rt_launcher
    assert "result->metric_count == paths->metric_count" in cuda_rt_launcher
    assert "params.geometry_mode == 2u" in cuda_rt_kernels
    assert "params.direct_first_hit" in cuda_rt_kernels
    assert "optixTerminateRay" in cuda_rt_kernels
    assert "OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT" in cuda_rt_kernels
    assert "membership_words_device" in cuda_rt_launcher
    assert "words_per_path" in cuda_rt_launcher
    assert "atomicOr" in cuda_rt_kernels
    assert "atomicAdd(&params.direct_inside_counts" in cuda_rt_kernels
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_kernels
    assert "score_decision_path_direct_stats_scatter_kernel" in cuda_rt_kernels
    assert "decision_path_target_stats_kernel" in cuda_rt_kernels
    assert "execute_decision_path_membership_sm" in cuda_rt_launcher
    assert "features_are_finite" in cuda_rt_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT" in cuda_rt_launcher
    assert "execute_decision_path_membership" in cuda_rt_launcher
    assert "gafime_gpu_decision_path_membership" in cuda_rt_launcher
    assert "gafime_gpu_decision_path_score" in cuda_rt_launcher
    assert "props.major == 8 && props.minor >= 9" in cuda_launcher
    assert "gafime_gpu_permutation_memory_peak" in cuda_launcher
    assert "gafime_gpu_permutation_pvalues" in cuda_launcher
    assert "gafime_gpu_decision_path_membership" not in cuda_launcher
    assert "gafime_gpu_decision_path_score" not in cuda_launcher
    assert "execute_decision_path_membership" not in cuda_launcher
    assert "execute_decision_path_score" not in cuda_launcher
    assert "inspect_cuda_matrix" in cuda_launcher
    assert "CudaMatrixView" in cuda_internal
    assert "has_covariance_metric" in cuda_launcher
    assert "if (has_covariance_metric(protocol))" in cuda_launcher
    assert "precision_has_covariance_metric" in rocm_launcher
    assert "if (precision_has_covariance_metric(protocol))" in rocm_launcher
    # CUDA keeps type-erased route dispatch in precision_kernels.cu and
    # specializes the histogram layout by bin count while sharing bounded
    # arity control. ROCm keeps bin hint validation and dispatch in its
    # launcher. Both must retain the complete ten-bin matrix.
    for launcher_text in (rocm_launcher,):
        assert "hint == 12" in launcher_text
        assert "hint == 24" in launcher_text
        assert "hint == 48" in launcher_text
    for launcher_text in (cuda_kernels, rocm_launcher):
        assert "case 12:" in launcher_text
        assert "case 24:" in launcher_text
        assert "case 48:" in launcher_text
    for marker in (
        "continuous_kernel",
        "mutual_info_kernel",
        "spearman_kernel",
        "select_topk_partials_kernel",
        "merge_topk_partials_kernel",
        "copy_selected_rows_kernel",
    ):
        assert marker in cuda_kernels
    for marker in (
        "score_continuous_chunk_kernel_static",
        "score_mutual_info_chunk_kernel_static",
        "score_spearman_chunk_kernel_static",
        "select_topk_partials_kernel_static",
        "merge_topk_partials_kernel_static",
        "copy_selected_rows_kernel",
    ):
        assert marker in rocm_kernels
    for device_text in (cuda_kernels, rocm_kernels, metal_shader):
        assert "previous_score" in device_text
        assert "previous_index" in device_text
        assert "already_selected" not in device_text
    assert "GAFIME_CUDA_PRECISION_INLINE" in cuda_kernels
    assert "GAFIME_HIP_FORCEINLINE" in rocm_kernels
    assert "hint == 16 || hint == 24 || hint == 32 || hint == 48" in metal_launcher
    assert "void compute_column_means_fp32(" in metal_launcher
    assert "sums[col] += features[base + col]" in metal_launcher
    assert "source_row == kInvalidIndex || source_row >= rank.row_count" in metal_shader
    assert "GAFIME_HIP_WAVE_MI_MASK 1" in rocm_kernels
    assert "kUseHipWaveMi" in rocm_kernels
    assert 'set(GAFIME_HIP_WAVE_MI_MODE "64" CACHE STRING' in rocm_cmake
    assert "off|64|96|64-96" in rocm_cmake
    assert (
        "MI_TEMPLATE_BIN_LEVELS: &[u32] = &[2, 4, 8, 12, 16, 24, 32, 48, 64, 96]"
        in continuous_combos
    )
    assert "pub const MI_SAMPLES_PER_JOINT_BIN: u64 = 8" in continuous_combos
    assert "select_adaptive_mi_bins_for_backend" in continuous_combos
    assert "MI_BINS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96)" in static_kernel_report
    assert "require-template-matrix" in static_kernel_report
    assert "shared runtime-arity body" in static_kernel_report
    assert "parameters[0] == 0" in static_kernel_report
    assert "require-topk-split" in static_kernel_report
    assert "require-no-spills" in static_kernel_report
    assert "require-precision-profiles" in static_kernel_report
    assert "precision_specialization_errors" in static_kernel_report
    assert "--verify-wheel-evidence" in static_kernel_report
    assert "wheel_sha256=" in static_kernel_report
    assert "native_sha256=" in static_kernel_report
    assert "require_precision_gathers=True" in static_kernel_report
    assert (
        'inspection_copy = temporary / "payload-inspection.so"' in static_kernel_report
    )
    assert "HIP static inspection mutated its input artifact" in static_kernel_report
    assert (
        "precision_kernel25copy_selected_rows_kernelIfEE" in static_kernel_report
    )
    assert (
        "precision_kernel25copy_selected_rows_kernelIdEE" in static_kernel_report
    )
    assert "__ockl_wfred_min_u32" in rocm_kernels
    assert "__ockl_wfred_max_u32" in rocm_kernels
    assert "__ockl_wfred_add_u32" in rocm_kernels
    assert (
        "reduce_mi[threadIdx.x] += reduce_mi[threadIdx.x + stride]"
        in rocm_kernels
    )
    assert "by_storage = 1 + (row_count - 1) / top_k" in cuda_launcher
    for launcher_text in (rocm_launcher, metal_launcher):
        assert "storage_blocks = 1 + (row_count - 1) / top_k" in launcher_text
    assert "std::sort(" not in rocm_launcher
    assert "std::stable_sort(" not in metal_launcher
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" not in cuda_launcher
    assert "direct_inside_counts" not in cuda_launcher
    assert "mix_permutation_seed" in cuda_launcher
    assert "0xA5A5A5A5" in cuda_launcher
    assert "--score-only" in cuda_rt_scale_bench
    assert "--direct-score" in cuda_rt_scale_bench
    assert "--firsthit-score" in cuda_rt_scale_bench
    assert "--bitset-score" in cuda_rt_scale_bench
    assert "--repeats=" in cuda_rt_scale_bench
    assert "timing repeats:" in cuda_rt_scale_bench
    assert "--throughput-only" in cuda_rt_scale_bench
    assert "--rt-only" in cuda_rt_scale_bench
    assert "--partitioned-grid" in cuda_rt_scale_bench
    assert "score parity      skipped (--throughput-only)" in cuda_rt_scale_bench
    assert "partitioned-grid" in cuda_rt_scale_bench
    assert "partitioned-grid direct scoring is throughput-only" in cuda_rt_scale_bench
    assert "--bitset-score/--firsthit-score" in cuda_rt_scale_bench
    assert "score-only default" in cuda_rt_scale_bench
    assert "--mixed-axes" in cuda_rt_scale_bench
    assert "--mixed-axis-pairs=" in cuda_rt_scale_bench
    assert "--overlap-axis-pairs=" in cuda_rt_scale_bench
    assert "mixed_axes && !score_only" in cuda_rt_scale_bench
    assert "box.feature0" in cuda_rt_scale_bench
    assert "box.feature1" in cuda_rt_scale_bench
    assert "time_cpu_score_boxes" in cuda_rt_scale_bench
    assert "validate_score_result" in cuda_rt_scale_bench
    assert "score_partitioned_grid_cpu" in cuda_rt_scale_bench
    assert "std::isfinite" in cuda_rt_scale_bench
    assert "gpu_rt_score_timing" in cuda_rt_scale_bench
    assert "firsthit work" in cuda_rt_scale_bench
    assert "ScoreResult gpu_rt_scores" in cuda_rt_scale_bench
    assert "std::vector<float> gpu_rt(output_len" in cuda_rt_scale_bench
    assert cuda_rt_scale_bench.index("if (score_only)") < cuda_rt_scale_bench.index(
        "std::vector<float> gpu_rt(output_len"
    )
    cuda_rt_firsthit_perf = (
        ROOT / "tests" / "release_measure" / "perf_05_cuda_rt_firsthit_scale.py"
    ).read_text()
    assert "GAFIME_CUDA_RT_SCALE_BENCH" in cuda_rt_firsthit_perf
    assert "GAFIME_CUDA_RT_FIRSTHIT_MIN_GEVALS" in cuda_rt_firsthit_perf
    assert "rt_max_abs" in cuda_rt_firsthit_perf
    assert "TIMING = re.compile" in cuda_rt_firsthit_perf
    assert "FIRSTHIT_WORK = re.compile" in cuda_rt_firsthit_perf
    assert "logical_work_denominator" in cuda_rt_firsthit_perf
    assert "--firsthit-score" in cuda_rt_firsthit_perf
    for paper_token in (
        "public \\gafime Python adapter now uses this compact score ABI",
        "0.347598",
        "118.077",
        "1.436361",
        "1.076311",
        "67.109",
        "rays/s",
        "PerfDigest",
        "docs/evidence/rt-firsthit-sm89-65536x8192-final.ncu-rep",
        "exact partition oracle",
        "matched partition index",
        "docs/evidence/rt-firsthit-hybrid-sm89-checkpoint.txt",
    ):
        assert paper_token in cuda_rt_paper
    for repro_token in (
        "narrow compact route",
        "gpu_rt_score_timing",
        "firsthit work",
        "PerfDigest MCP",
        "compare_metrics(",
        "5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331",
        "O(rows * groups + paths)",
        "SOURCE_DATE_EPOCH=1784592000",
    ):
        assert repro_token in cuda_rt_paper_repro
    profiler_evidence = (
        ROOT / "docs" / "evidence" / "rt-firsthit-sm89-65536x8192-final.ncu-rep"
    )
    assert profiler_evidence.stat().st_size == 31_848_275
    assert (
        hashlib.sha256(profiler_evidence.read_bytes()).hexdigest()
        == "5461bf86495d9a12666891bba2f334ecea8b16b3c8cb806168a557101a52c331"
    )
    assert (
        "perf_05_cuda_rt_firsthit_scale.py"
        in (ROOT / "tests" / "release_measure" / "run_gpu_suite.sh").read_text()
    )

    assert "namespace kernel" in rocm_header
    assert "TargetStatsDevice" in rocm_header
    assert "UnaryFeatureStatsDevice" in rocm_header
    assert "launch_precision_target_stats" in rocm_launcher
    assert "launch_precision_unary_feature_stats" in rocm_launcher
    assert "launch_continuous_chunk" in rocm_header
    assert "launch_mutual_info_chunk" in rocm_header
    assert "launch_spearman_chunk" in rocm_header
    # CUDA's canonical route engine deliberately removed the duplicate legacy
    # declaration tree. Its private header is now one profile-specialized
    # operation table; concrete typed stats and kernels live in the owning TU.
    assert "namespace kernel" not in cuda_header
    assert "struct CudaPrecisionKernelSet" in cuda_precision_header
    for operation in (
        "(*target_stats)",
        "(*feature_stats)",
        "(*continuous)",
        "(*mutual_info)",
        "(*spearman)",
        "(*selected_metric_max)",
        "(*accumulate_exceedances)",
    ):
        assert operation in cuda_precision_header
    assert "TargetStatsDevice" in cuda_kernels
    assert "UnaryFeatureStatsDevice" in cuda_kernels
    assert "launch_decision_path_membership" not in cuda_header
    assert "decision_path_membership_kernel" in cuda_rt_header
    assert "GafimeRtBox" in cuda_rt_header
    assert "GafimeRtTriVertex" in cuda_rt_header
    assert "GafimeRtTriIndex" in cuda_rt_header
    assert "pack_decision_path_points_kernel" in cuda_rt_header
    assert "decision_path_bitset_kernel" in cuda_rt_header
    assert "score_decision_path_bitset_kernel" in cuda_rt_header
    assert "score_decision_path_direct_stats_kernel" in cuda_rt_header
    assert "score_decision_path_direct_stats_scatter_kernel" in cuda_rt_header
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
    assert (
        "GAFIME_METRIC_MUTUAL_INFO" in metal_launcher
        and "GAFIME_METRIC_SPEARMAN" in metal_launcher
    )
    assert "launcher.mm" in metal_cmake and "shader.metal" in metal_cmake
    assert "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>" in metal_cmake
    assert "xcrun" in metal_cmake and "-fno-fast-math" in metal_cmake
    metal_mi_body = metal_shader.split(
        "kernel void gafime_score_mutual_info(", 1
    )[1].split("kernel void gafime_build_spearman_target_ranks(", 1)[0]
    assert "float local_mi =" not in metal_mi_body
    assert "s_uint1" not in metal_mi_body
    assert "for (uint xb = 0; xb < bins; ++xb)" in metal_mi_body
    assert "for (uint yb = 0; yb < bins; ++yb)" in metal_mi_body
    assert "bool content_valid;" in metal_launcher
    assert "matrix->content_valid = false;" in metal_launcher
    assert "if (!matrix->content_valid ||" in metal_launcher

    cuda_execute_body = cuda_launcher.split(
        "int execute_precision(\n", 1
    )[1].split("uint64_t resident_precision_bytes", 1)[0]
    assert cuda_execute_body.index("validate_precision_protocol") < cuda_execute_body.index(
        "require_precision_matrix(matrix)"
    )
    assert cuda_execute_body.index("validate_precision_result_") < cuda_execute_body.index(
        "require_precision_matrix(matrix)"
    )
    for name, launcher_text, protocol_marker, result_marker, content_marker in (
        (
            "rocm",
            rocm_launcher,
            "validate_protocol",
            "validate_result_table",
            "require_valid_matrix_content(matrix)",
        ),
        (
            "metal",
            metal_launcher,
            "validate_protocol",
            "validate_result_table",
            "if (!matrix->content_valid ||",
        ),
    ):
        execute_body = launcher_text.split(
            "GAFIME_GPU_API int gafime_gpu_execute(\n", 1
        )[1]
        assert execute_body.index(protocol_marker) < execute_body.index(
            content_marker
        ), f"{name} checks content before protocol ABI"
        assert execute_body.index(result_marker) < execute_body.index(
            content_marker
        ), f"{name} checks content before result ABI"

    cuda_membership_body = cuda_rt_launcher.split(
        "GAFIME_GPU_API int gafime_gpu_decision_path_membership", 1
    )[1]
    assert cuda_membership_body.index(
        "paths->abi_version"
    ) < cuda_membership_body.index("inspect_cuda_matrix(matrix_handle, &matrix)")
    assert "GAFIME_MAX_DECISION_PATH_COUNT" in cuda_rt_launcher

    assert "defined(GAFIME_BUILDING_DLL)" in cuda_api
    assert "#define GAFIME_GPU_BUILDING_DLL" in cuda_api
    assert "GAFIME_GPU_BUILDING_DLL" in cuda_cmake
    assert "-DGAFIME_BUILDING_DLL" not in stage_gpu_payload
    assert stage_gpu_payload.count("-DGAFIME_GPU_BUILDING_DLL") == 2
    assert stage_gpu_payload.count("-DGAFIME_GPU_MI_ACCUMULATION_FP64=0") == 2
    assert '"covariance_policy.hpp"' in stage_gpu_payload

    assert "build_host_resident" in cuda_launcher
    assert "resident.data()" in cuda_launcher
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
    ):
        assert marker in common_header, marker

    for marker in (
        "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT",
        "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT",
    ):
        assert marker not in common_header, marker
        assert marker in cuda_rt_abi, marker

    assert "cuda_architecture_class" in cuda_launcher
    assert "precision_cuda_device_flags" in cuda_launcher
    assert "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT" not in cuda_launcher
    assert "GAFIME_CUDA_ENABLE_OPTIX_RT" not in cuda_launcher
    assert "GAFIME_CUDA_LOCAL_DEVICE_FLAGS" in cuda_launcher
    assert "cudaDriverGetVersion" in cuda_launcher
    assert "cudaRuntimeGetVersion" in cuda_launcher
    assert "tune_rt_kernels_for_device" not in cuda_launcher
    assert "tune_rt_kernels_for_device" in cuda_rt_launcher

    assert "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES" in optix_smoke
    assert "__intersection__gafime_dp_box" in optix_smoke
    assert "__anyhit__gafime_dp_mark" in optix_smoke
    assert "open_lo_mask" in optix_smoke
    assert "sm_decision_path_membership_kernel" in optix_smoke
    assert "GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP" in cuda_abi_smoke
    assert "GAFIME_CUDA_TEST_LOCAL_RT" in cuda_abi_smoke
    assert "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT" in cuda_abi_smoke
    assert "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT" in cuda_abi_smoke
    assert "gafime_gpu_decision_path_score" in cuda_abi_smoke

    assert "gcnArchName" in rocm_launcher
    assert "rocm_arch_is_rdna" in rocm_launcher
    assert "rocm_arch_is_cdna" in rocm_launcher
    assert "rocm_use_managed_memory" in rocm_launcher
    assert "hipMallocManaged" in rocm_launcher
    assert (
        "props.managedMemory != 0 || props.concurrentManagedAccess != 0"
        in rocm_launcher
    )
    assert "return hip_status(hipMallocManaged" in rocm_launcher
    assert "return props.integrated != 0;" in rocm_launcher
    assert "ascii_contains_ci" not in rocm_launcher
    for product_marker in ("Radeon Graphics", "APU", "iGPU", "780M", "870M"):
        assert product_marker not in rocm_launcher

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

    assert "precision_kernels.cu" in cuda_cmake
    assert "precision_launcher.cu" in cuda_cmake
    assert "\n    kernels.cu" not in cuda_cmake
    assert "\n    launcher.cu" not in cuda_cmake
    assert "rt_kernels.cu" in cuda_cmake and "rt_launcher.cu" in cuda_cmake
    assert "GAFIME_CUDA_ENABLE_OPTIX_RT" in cuda_cmake
    assert "GAFIME_CUDA_RT_BUILD_MODE" in cuda_cmake
    assert "^(off|on|both)$" in cuda_cmake
    assert "GAFIME_CUDA_MI_ACCUMULATION_MODE" not in cuda_cmake
    for profile in (
        "GAFIME_PRECISION_FP32",
        "GAFIME_PRECISION_MIXED",
        "GAFIME_PRECISION_FP64",
    ):
        assert profile in cuda_precision_kernels
        assert profile in cuda_precision_launcher
    assert "PrecisionTraits" in cuda_precision_kernels
    assert "CudaPrecisionKernelSet" in cuda_precision_header
    cuda_score_body = cuda_precision_launcher.split(
        "int launch_precision_score_kernels(", 1
    )[1].split("uint64_t metric_signature(", 1)[0]
    assert (
        "const uint64_t* chunk_cached_ranks = chunk.arity == 1 ? cached_ranks : nullptr;"
        in cuda_score_body
    )
    assert cuda_score_body.count("chunk_cached_ranks") == 2

    cuda_mixed_traits = cuda_precision_kernels.split(
        "struct PrecisionTraits<GAFIME_PRECISION_MIXED>", 1
    )[1].split("struct PrecisionTraits<GAFIME_PRECISION_FP64>", 1)[0]
    assert "using Storage = float;" in cuda_mixed_traits
    assert "using Accumulation = double;" in cuda_mixed_traits
    cuda_mi_body = cuda_precision_kernels.split(
        "__global__ void mutual_info_kernel(", 1
    )[1].split("rank_twice_for_value(", 1)[0]
    assert "const Storage inverse_x" in cuda_mi_body
    assert "const Storage inverse_y" in cuda_mi_body
    assert "fixed_mi_bin(raw_x, minimum_x, inverse_x, effective_bins)" in cuda_mi_body
    assert "fixed_mi_bin(raw_y, minimum_y, inverse_y, effective_bins)" in cuda_mi_body
    assert "const Accumulation inverse_x" not in cuda_mi_body
    assert "const Accumulation inverse_y" not in cuda_mi_body
    assert "gafime_cuda_v1_rt" in cuda_cmake
    assert "GAFIME_CUDA_TUNING_SM" not in cuda_cmake
    assert "CudaKernelLaunchPolicy" in cuda_header
    assert "cuda_kernel_launch_policy_for_device" in cuda_header
    assert "props.maxThreadsPerBlock" in cuda_launcher
    assert "matrix->launch_policy = launch_policy" in cuda_launcher
    assert "cuda_spearman_target_cache_bench.cpp" in cuda_cmake
    assert 'CUDA_TUNING_POLICY = "runtime-device-class"' in stage_gpu_payload
    assert "RUNTIME_ARCHITECTURE_DISPATCH = True" in stage_gpu_payload
    assert "--ptx" in cuda_cmake and "rt_kernels.cu" in cuda_cmake
    assert "gafime_rt_optix_ptx.hpp" in cuda_cmake
    assert "CUDA::cuda_driver" in cuda_cmake
    assert "kernels.hip" in rocm_cmake and "launcher.hip" in rocm_cmake
    assert "GAFIME_HIP_MI_ACCUMULATION_MODE" not in rocm_cmake
    assert "GAFIME_HIP_PRECISION_PROFILE_MASK 7" in rocm_cmake
    assert "gafime_rocm_native_timing" in rocm_cmake
    assert "historical pre-freeze typed precision ABI 1.1 baseline" in rocm_cmake
    for rocm_native_surface in (
        "FrozenPrecisionMatrixDesc",
        "FrozenPrecisionCapabilities",
        "FrozenPrecisionLaunchProtocol",
        "FrozenResultTableF64",
        "FrozenInteractionDiagnosticBatch",
        "gafime_gpu_precision_capabilities",
        "gafime_gpu_numeric_routes_v2",
        "gafime_gpu_permutation_memory_peak_v2",
        "gafime_gpu_permutation_pvalues_v2",
        "gafime_gpu_interaction_diagnostics_v2",
        "gafime_gpu_interaction_diagnostics",
        "gafime_gpu_matrix_upload_f32_v2",
        "gafime_gpu_matrix_upload_f64_v2",
        "gafime_gpu_execute_f32_v2",
        "gafime_gpu_execute_f64_v2",
        "gafime_gpu_execution_memory_peak_v2",
        "precision-typed-v1.1",
        "numeric-route-v2",
        "canonical_payload_resolution",
        "optional_symbols_missing",
        "optional_permutation_status",
        "representative_direct_transfer",
        "target_update",
        "harness_source_root",
        "wrapper_comparability",
        "execution_memory_forecast",
    ):
        assert rocm_native_surface in rocm_native_timing
    for rocm_native_doc_marker in (
        "numeric-route-v2",
        "precision-typed-v1.1",
        "optional present/missing metadata",
        "representative_direct_transfer",
        "common helper/harness provenance",
    ):
        assert rocm_native_doc_marker in rocm_native_doc
    assert "PrecisionLane::Fp32" in rocm_precision
    assert "PrecisionLane::Mixed" in rocm_precision
    assert "PrecisionLane::Fp64" in rocm_precision
    rocm_mixed_traits = rocm_precision.split(
        "struct PrecisionTraits<PrecisionLane::Mixed>", 1
    )[1].split("struct PrecisionTraits<PrecisionLane::Fp64>", 1)[0]
    assert "using Storage = float;" in rocm_mixed_traits
    assert "using Accumulator = double;" in rocm_mixed_traits
    rocm_precision_kernels = rocm_kernels.split("namespace precision_kernel {", 1)[1]
    rocm_mi_body = rocm_precision_kernels.split(
        "__global__ void score_mutual_info_chunk_kernel_static(", 1
    )[1].split("__global__ void score_spearman_chunk_kernel_static(", 1)[0]
    assert "const StorageT inv_x" in rocm_mi_body
    assert "const StorageT inv_y" in rocm_mi_body
    assert rocm_mi_body.count("precision_fixed_mi_bin<StorageT>(") >= 2
    assert "x, min_x, inv_x, Bins" in rocm_mi_body
    assert "y, min_y, inv_y, Bins" in rocm_mi_body
    assert "precision_fixed_mi_bin<AccumT>" not in rocm_mi_body
    assert (
        "GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(float, float, float)" in rocm_kernels
    )
    assert (
        "GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(float, double, double)"
        in rocm_kernels
    )
    assert (
        "GAFIME_HIP_INSTANTIATE_PRECISION_PROFILE(double, double, double)"
        in rocm_kernels
    )

    for integer_metal_rank_surface in (
        "device uint* target_ranks_twice [[buffer(1)]]",
        "device const uint* target_ranks_twice [[buffer(8)]]",
        "uint less_x = 0;",
        "uint eq_x = 0;",
        "uint less_y = 0;",
        "uint eq_y = 0;",
        "const uint rx_twice",
        "const uint ry_twice",
        "uint local_valid = 0;",
        "threadgroup uint s_uint0[kMetalReduceWidth]",
        "const uint valid = s_uint0[0];",
        "ulong l_n = 0;",
        "threadgroup ulong s_n[kMetalReduceWidth]",
    ):
        assert integer_metal_rank_surface in metal_shader
    assert "matrix_desc->rows * sizeof(uint32_t)" in metal_launcher
    for forbidden_metal_float_counter in (
        "device float* target_ranks_twice",
        "device const float* target_ranks_twice",
        "float less_x",
        "float eq_x",
        "float less_y",
        "float eq_y",
        "float local_valid",
        "threadgroup float s_n",
    ):
        assert forbidden_metal_float_counter not in metal_shader

    correlation_f32 = covariance_f32_text.split("fn finish(self) -> f32", 1)[1].split(
        "struct EqualVectorParts", 1
    )[0].replace("self.", "")
    correlation_f64 = cpu_precision_kernels.split("fn finalize_correlation_f64", 1)[
        1
    ].split("fn finalize_r2_f32", 1)[0]
    for finalizer, nan_type in (
        (correlation_f32, "f32::NAN"),
        (correlation_f64, "f64::NAN"),
    ):
        assert "!variance_x.is_finite()" in finalizer
        assert "!variance_y.is_finite()" in finalizer
        assert "!covariance.is_finite()" in finalizer
        assert "!denominator.is_finite()" in finalizer
        assert "correlation.is_finite()" in finalizer
        assert nan_type in finalizer
    assert (
        "correlation_finalization_preserves_arithmetic_failure_as_nan"
        in cpu_precision_kernels
    )

    rank_value_body = rust_precision_ranking.split("fn rank_value_at(", 1)[1].split(
        "enum PrecisionRankValue", 1
    )[0]
    assert rank_value_body.count(".filter(|value| value.is_finite())") == 2
    assert rank_value_body.count(".filter(|(_, value)| value.is_finite())") == 2
    assert "if rank_value_at(table, metric_ids, row, metric_index).is_none()" in (
        rust_precision_ranking
    )
    assert "fp32_ranking_excludes_nonfinite_public_scores" in rust_precision_ranking
    assert "f64_ranking_excludes_nonfinite_public_scores" in rust_precision_ranking
    assert "range_to_f64" not in rust_public_precision
    assert "precision_values_to_public" not in rust_public_precision
    assert "pvalues: Vec<f64>" not in rust_public_precision
    assert "means: Vec<f64>" not in rust_public_precision
    assert "stds: Vec<f64>" not in rust_public_precision
    assert "precision_values_to_python_array" in rust_public_precision
    assert 'array.call1(("f", values))' in rust_public_precision
    assert 'array.call1(("d", values))' in rust_public_precision
    assert 'py.import("numpy")' not in rust_public_precision
    assert "support: int = 0" in python_decision_path
    assert '"support": self.support' in python_decision_path

    for metal_profile_identity in (
        "descriptor_profile",
        "feature_stats_profile",
        "target_stats_profile",
        "spearman_target_ranks_profile",
    ):
        assert metal_profile_identity in metal_launcher
    assert "matrix->descriptor_profile == matrix->precision_profile" in metal_launcher
    assert (
        "matrix->feature_stats_profile != matrix->precision_profile" in metal_launcher
    )
    assert "matrix->target_stats_profile != matrix->precision_profile" in metal_launcher
    assert (
        "matrix->spearman_target_ranks_profile != matrix->precision_profile"
        in metal_launcher
    )
    legacy_metal_means = metal_launcher.split(
        "void compute_column_means_legacy(", 1
    )[1].split("void compute_column_means_fp32(", 1)[0]
    fp32_metal_means = metal_launcher.split(
        "void compute_column_means_fp32(", 1
    )[1].split("void build_feature_major(", 1)[0]
    metal_upload = metal_launcher.split(
        "GAFIME_GPU_API int gafime_gpu_matrix_upload(", 1
    )[1].split("GAFIME_GPU_API int gafime_gpu_matrix_update_target(", 1)[0]
    assert "std::vector<double> sums(cols, 0.0);" in legacy_metal_means
    assert "std::vector<float> sums(cols, 0.0f);" in fp32_metal_means
    assert "std::vector<double>" not in fp32_metal_means
    assert "matrix->precision_profile == GAFIME_PRECISION_FP32" in metal_upload
    assert "compute_column_means_fp32(features_host, rows, cols, means);" in metal_upload
    assert "compute_column_means_legacy(features_host, rows, cols, means);" in metal_upload
    assert "matrix->precision_profile = 0;" in metal_launcher
    assert (
        "matrix->precision_profile = GAFIME_PRECISION_FP32;" in metal_launcher
    )

    assert 'backend="metal",\n              precision="fp32",' in metal_lab_workflow
    assert "precision_01_end_to_end_profiles.py" in native_validation_workflow
    assert "--backend metal" in native_validation_workflow
    assert "tests/release_measure/perf_13_precision_profiles.py" in metal_beast_workflow
    assert "tests/release_measure/perf_12_precision_profiles.py" not in metal_beast_workflow
    assert "--backend metal" in metal_beast_workflow
    assert "--profile fp32" in metal_beast_workflow
    assert "--warmups 10" in metal_beast_workflow
    assert "--repetitions 30" in metal_beast_workflow
    assert "gafime.precision-profile-native-evidence.v1" in metal_beast_workflow
    assert "gafime.metal.native_timing.v1" in metal_beast_workflow
    assert "gpu_timing_supported" in metal_beast_workflow
    assert "provenance_hashes_verified" in metal_beast_workflow
    assert "GAFIME_METAL_BUILD_BENCHMARKS" in metal_cmake
    assert "metal_precision_native_timing.mm" in metal_cmake
    assert "gafime_metal_precision_native_timing" in metal_cmake
    assert "command_buffer.GPUStartTime" in metal_native_timing
    assert "command_buffer.GPUEndTime" in metal_native_timing
    assert "[command_buffer commit];" in metal_native_timing
    assert "[command_buffer waitUntilCompleted];" in metal_native_timing
    assert metal_native_timing.index("[command_buffer waitUntilCompleted];") < (
        metal_native_timing.index("command_buffer.GPUStartTime")
    )
    assert "kDefaultWarmups = 10" in metal_native_timing
    assert "kDefaultRepeats = 30" in metal_native_timing
    assert "_validate_metal_native_timing_artifact" in precision_profile_perf
    assert "complete_gpu_timestamp_support_required" in precision_profile_perf
    assert "gafime_gpu_numeric_routes_v2" in metal_native_timing
    assert "gafime_gpu_matrix_alloc_v2" in metal_native_timing
    assert "gafime_gpu_matrix_upload_v2" in metal_native_timing
    assert "gafime_gpu_matrix_update_target_v2" in metal_native_timing
    assert "gafime_gpu_execute_v2" in metal_native_timing
    assert "gafime_gpu_execution_memory_peak_v2" in metal_native_timing
    assert "gafime_gpu_permutation_memory_peak_v2" in metal_native_timing
    assert "gafime_gpu_permutation_pvalues_v2" in metal_native_timing
    assert "gafime_gpu_interaction_diagnostics_v2" in metal_native_timing
    assert "gafime_gpu_matrix_free_v2" in metal_native_timing
    for metal_baseline_surface in (
        "FrozenPrecisionMatrixDesc",
        "FrozenPrecisionCapabilities",
        "FrozenPrecisionLaunchProtocol",
        "precision-typed-v1.1",
        "numeric-route-v2",
        "gafime_gpu_precision_capabilities",
        "gafime_gpu_matrix_upload_f32_v2",
        "gafime_gpu_execute_f32_v2",
    ):
        assert metal_baseline_surface in metal_native_timing
    assert '\\"source_tree_state\\"' in metal_native_timing
    assert '\\"product_source_tree_state\\"' in metal_native_timing
    assert '\\"harness_source_blob\\"' in metal_native_timing
    assert 'write_file_identity(output, "harness_source", source_path, true)' in (
        metal_native_timing
    )
    assert "const ProductSourceBinding product_source = bind_product_source(" in (
        metal_native_timing
    )
    assert "options.shader_source_path" not in metal_native_timing.split(
        "const ProductSourceBinding product_source = bind_product_source(", 1
    )[1].split("const SourceBinding harness_source", 1)[0]
    assert '\\"input_policy\\"' in metal_native_timing
    assert '\\"input_identity\\"' in metal_native_timing
    assert '\\"environment\\"' in metal_native_timing
    assert '\\"clock_and_power_capture_point\\"' in metal_native_timing
    assert '\\"clock_and_power_state\\"' in metal_native_timing
    assert "system_profiler SPDisplaysDataType -json" in metal_native_timing
    assert "pmset -g custom" in metal_native_timing
    assert "no dynamic GPU metric" in metal_native_timing
    assert "8dfcd20a148d90917f654aea707c324535474b0a" in metal_beast_workflow
    expected_candidate_input = metal_beast_workflow.split(
        "expected_candidate_sha:", 1
    )[1].split("preset:", 1)[0]
    assert "required: true" in expected_candidate_input
    assert "type: string" in expected_candidate_input
    assert "ref: ${{ github.event.inputs.expected_candidate_sha }}" in (
        metal_beast_workflow
    )
    assert "EXPECTED_CANDIDATE_SHA: ${{ github.event.inputs.expected_candidate_sha }}" in (
        metal_beast_workflow
    )
    assert "expected_candidate_sha must be a full lowercase commit SHA" in (
        metal_beast_workflow
    )
    assert 'f"{os.environ[\'GITHUB_API_URL\']}/repos/{repository}/pulls/70"' in (
        metal_beast_workflow
    )
    assert 'live_head.get("ref") != "codex/issue-25-three-precision-profiles"' in (
        metal_beast_workflow
    )
    assert "live PR #70 head moved during cold evidence" in metal_beast_workflow
    assert '"schema": "gafime.hosted-exact-pr-head.v1"' in metal_beast_workflow
    assert 'exact_head["post_evidence_status"] = "pass"' in metal_beast_workflow
    assert '--source-root "$product_root"' in metal_beast_workflow
    assert '--harness-source-root "$GITHUB_WORKSPACE"' in metal_beast_workflow
    assert '--variant baseline="$METAL_BASELINE_VENV/bin/python"' in (
        metal_beast_workflow
    )
    assert '--variant candidate="$METAL_VENV/bin/python"' in metal_beast_workflow
    assert '--native-evidence baseline=' in metal_beast_workflow
    assert '--native-evidence candidate=' in metal_beast_workflow
    assert "--ab-blocks 2" in metal_beast_workflow
    assert "-DGAFIME_ABI_CONSUMER_ENABLE_TYPED_BASELINE=ON" in metal_beast_workflow
    assert "installed-payload-baseline/gafime/_metal/libgafime_metal_v1.dylib" in (
        metal_beast_workflow
    )
    assert "gafime_abi_1_1_typed_c_consumer_metal_baseline" in metal_beast_workflow
    assert "workload_args=(--workload release)" in metal_beast_workflow
    assert "ab_schedule_readiness" in metal_beast_workflow
    assert "native_ab_schedule_readiness" in metal_beast_workflow
    assert "fresh_helper_process_per_variant_trial" in metal_beast_workflow
    assert "for input_policy in common-f64 native; do" in metal_beast_workflow
    assert '--input-policy "$input_policy"' in metal_beast_workflow
    assert "metal-native-timing-$input_policy-block-" in metal_beast_workflow
    assert "assert len(artifacts) == 8" in metal_beast_workflow
    assert '"input_policy": input_policy' in metal_beast_workflow
    assert '"common-f64", "native"' in metal_beast_workflow
    assert '--ab-block "$ab_block"' in metal_beast_workflow
    assert '--variant-sequence "$variant_sequence"' in metal_beast_workflow
    assert '\\"ab_block\\"' in metal_native_timing
    assert '\\"variant_sequence\\"' in metal_native_timing
    assert "fresh_helper_process_per_variant_trial" in metal_native_timing
    assert "comparative_input_readiness" in metal_beast_workflow
    assert 'lifecycle["fp64_route_rejected"] is True' in metal_beast_workflow
    assert 'for precision in ("mixed", "fp64")' in metal_beast_workflow
    assert "installed-wheel explicit Metal mixed/fp64 rejection: pass" in (
        metal_beast_workflow
    )
    assert 'cold_harness="$GITHUB_WORKSPACE/tests/release_measure/cold_lifecycle.py"' in (
        metal_beast_workflow
    )
    assert 'cold_harness_blob="$(git hash-object "$cold_harness")"' in (
        metal_beast_workflow
    )
    assert 'cold_head_blob="$(git rev-parse "HEAD:$cold_relative")"' in (
        metal_beast_workflow
    )
    assert 'variant_order=(baseline candidate)' in metal_beast_workflow
    assert 'variant_order=(candidate baseline)' in metal_beast_workflow
    assert '"$METAL_BASELINE_VENV/bin/python"' in metal_beast_workflow
    assert '"$METAL_VENV/bin/python"' in metal_beast_workflow
    assert '"$python_executable" "$cold_harness"' in metal_beast_workflow
    assert "--compare-manifest" in metal_beast_workflow
    assert '"schema": "gafime.cold-lifecycle-comparison-manifest.v1"' in (
        metal_beast_workflow
    )
    assert '"precision-typed-v1.1"' in metal_beast_workflow
    assert '"numeric-route-v2"' in metal_beast_workflow
    assert 'comparison["valid_for_canonical_cold_lifecycle_claims"] is True' in (
        metal_beast_workflow
    )
    assert 'comparison["provenance"]["baseline_commit"]' in metal_beast_workflow
    assert 'comparison["provenance"]["candidate_commit"]' in metal_beast_workflow
    assert "30 fresh subprocesses per " in metal_beast_workflow
    assert "metal-cold-lifecycle-comparison-" in metal_beast_workflow
    assert "canonical_payload_records" in metal_native_timing
    assert "core_release_benchmark_is_an_external_common_harness" in core_native_timing
    assert "precision_profiles_native_release_benchmark" not in core_native_timing
    assert "cargo test" not in core_native_runner
    assert '"--extern"' in core_native_runner
    assert 'f"gafime_cpu={product_rlib}"' in core_native_runner
    assert "_tracked_source_identity" in core_native_runner
    assert "_compiler_environment" in core_native_runner
    for core_provenance_marker in (
        "GAFIME_NATIVE_PRODUCT_SOURCE_ROOT",
        "GAFIME_NATIVE_HARNESS_SOURCE_ROOT",
        "GAFIME_NATIVE_HARNESS_SOURCE_SHA256",
        "GAFIME_NATIVE_HARNESS_SOURCE_GIT_BLOB",
        "GAFIME_COMPILED_HARNESS_RUNNER_SHA256",
        "GAFIME_NATIVE_PRODUCT_RLIB",
        "GAFIME_NATIVE_PRODUCT_RLIB_SHA256",
        "GAFIME_NATIVE_BENCH_BINARY_SHA256",
        "product_source_commit",
        "product_source_tree",
        "harness_source_commit",
        "harness_source_tree",
        "harness_source_blob",
        "harness_runner_blob",
        "current_git_blob",
        "head_git_blob",
        "order_position_median_ns",
        "investigate_possible_order_contamination",
        "pearson_f32",
        "spearman_f32",
        "mutual_info_fixed_f32",
        "command_argv",
        "policy_clock_state",
        "platform_power_profile",
        "repeatable_contamination_cells",
    ):
        assert core_provenance_marker in core_native_standalone
    cuda_environment = cuda_native_timing.split(
        "std::vector<std::string> observed_environment()", 1
    )[1].split("std::vector<std::string> values", 1)[0]
    assert '"PATH"' in cuda_environment
    rocm_environment = rocm_native_timing.split(
        "void append_environment_json(std::ostringstream& stream)", 1
    )[1].split("stream << '{';", 1)[0]
    for environment_key in (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "RAYON_NUM_THREADS",
    ):
        assert f'"{environment_key}"' in rocm_environment
    assert "installed_payload_dylib" in precision_profile_perf
    assert "canonical_payload_records" in precision_profile_perf
    assert "installed_payload_root" in metal_beast_workflow
    assert "heterogeneous_backend_profile_matrices_require_separate_runs" in (
        precision_profile_perf
    )
    assert "gafime.cold-lifecycle.v1" in cold_lifecycle
    assert "gafime_gpu_numeric_routes_v2" in cold_lifecycle
    assert "gafime_gpu_matrix_alloc_v2" in cold_lifecycle
    assert "gafime_gpu_matrix_upload_v2" in cold_lifecycle
    assert "gafime_gpu_matrix_update_target_v2" in cold_lifecycle
    assert "gafime_gpu_execution_memory_peak_v2" in cold_lifecycle
    assert "gafime_gpu_execute_v2" in cold_lifecycle
    assert "gafime_gpu_matrix_free_v2" in cold_lifecycle
    assert '"cudaFree"' in cold_lifecycle
    assert '"hipFree"' in cold_lifecycle
    assert '"observed_combined"' in cold_lifecycle
    assert "args.repetitions < 30" in cold_lifecycle
    assert "tests/metal_hardcore_benchmark.py" not in metal_beast_workflow
    assert '"$METAL_VENV/bin/python" -m pip install numpy "$METAL_WHEELHOUSE"/gafime-*.whl' in (
        metal_beast_workflow
    )
    assert '"$METAL_BASELINE_VENV/bin/python" -m pip install numpy "$METAL_BASELINE_WHEELHOUSE"/gafime-*.whl' in (
        metal_beast_workflow
    )
    assert (
        "python -m pip install numpy wheelhouse/gafime-*.whl"
        in native_validation_workflow
    )
    assert "GAFIME_CUDA_MI_ACCUMULATION_MODE" not in contract_workflow
    assert "gafime_cuda_precision_abi_smoke" in cuda_cmake
    assert "gafime_rocm_precision_abi_smoke" in rocm_cmake

    assert 'precision: str = "mixed"' in python_config
    assert "storage_dtype: str" not in python_config
    assert "compute_policy: str" not in python_config
    assert 'SUPPORTED_PRECISIONS = ("fp32", "mixed", "fp64")' in python_precision
    assert "precision_from_legacy_pair" in python_precision
    assert "backend_precision_error" in python_adapter
    assert "precision_contract: CapabilityValue" in python_capabilities
    assert 'get_string(config, "precision", "mixed")' in rust_runtime
    assert 'runtime.set_item("precision", precision)' in rust_runtime
    assert "GAFIME_DTYPE_F64 = 2" in precision_doc
    assert 'EngineConfig(precision="mixed")' in precision_doc
    assert "Core | yes | yes | yes" in precision_doc
    assert "Metal | yes | no | no" in precision_doc
    assert "no intermediate fp32 quantization" in precision_doc
    assert "array('f')" in precision_doc
    assert "array('d')" in precision_doc
    assert "## Interaction Materialization Diagnostics" in precision_doc
    assert "ordinary path does not add a second matrix-row scan" in precision_doc
    assert "docs/evidence/interaction-overflow-diagnostics.md" in precision_doc
    assert "Widened or" in precision_doc
    assert "log-domain interaction evaluation" in precision_doc
    assert "not a universal performance claim" in interaction_diagnostic_evidence
    assert "candidate-minus-base steady resident" in interaction_diagnostic_evidence
    assert "Metal was not compiled or timed locally" in interaction_diagnostic_evidence
    assert (
        "docs/evidence/cpu-covariance-finite-pass.md"
        in (ROOT / "docs" / "cpu-fused-continuous-accumulation.md").read_text()
    )
    assert "bounded local observation" in cpu_covariance_evidence
    assert "first-row NaN" in cpu_covariance_evidence
    assert "not a release-wide throughput guarantee" in cpu_covariance_evidence
    assert (
        "docs/evidence/cpu-mi-branchless-bins.md"
        in (ROOT / "docs" / "cpu-fused-continuous-accumulation.md").read_text()
    )
    assert "bounded local" in cpu_mi_evidence
    assert "larger helper ratios are not public MI throughput claims" in cpu_mi_evidence
    assert "No unchecked scatter was added" in cpu_mi_evidence
    assert "8 * args.bins * args.bins" in cpu_mi_perf


def check_native_abi_and_reduce_scale_structure() -> None:
    types_text = (ROOT / "crates" / "gafime-types" / "src" / "lib.rs").read_text()
    local_types_text = (
        ROOT / "crates" / "gafime-types" / "src" / "local_cmake_experiment.rs"
    ).read_text()
    cuda_rt_abi = (ROOT / "src" / "cuda" / "rt_abi.hpp").read_text()
    covariance_policy = (ROOT / "src" / "common" / "covariance_policy.hpp").read_text()
    cuda_launcher = (ROOT / "src" / "cuda" / "precision_launcher.cu").read_text()
    cuda_kernels = (ROOT / "src" / "cuda" / "precision_kernels.cu").read_text()
    rocm_launcher = (ROOT / "src" / "rocm" / "launcher.hip").read_text()
    rocm_kernels = (ROOT / "src" / "rocm" / "kernels.hip").read_text()
    metal_launcher = (ROOT / "src" / "metal" / "launcher.mm").read_text()
    metal_shader = (ROOT / "src" / "metal" / "shader.metal").read_text()
    assert "include_str!(" in types_text
    assert "src/common/gafime_gpu_abi.hpp" in types_text
    assert "gpu_abi_header_and_rust_layouts_stay_in_lockstep" in types_text
    assert "offset_of!(GafimeLaunchProtocol, permutations)" in types_text
    assert "offset_of!(GafimeResultTable, backend_private)" in types_text
    assert "GafimePermutationSignificanceTable" in types_text
    for local_type in (
        "GafimeDecisionPathTerm",
        "GafimeDecisionPathBatch",
        "GafimeDecisionPathScoreBatch",
    ):
        assert local_type not in types_text
        assert local_type in local_types_text
    assert "src/cuda/rt_abi.hpp" in local_types_text
    assert "GafimeInteractionDiagnosticBatch" in types_text
    assert (
        "gafime_gpu_permutation_memory_peak"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    assert (
        "gafime_gpu_permutation_pvalues"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    common_header = (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    assert "gafime_gpu_decision_path_membership" not in common_header
    assert "gafime_gpu_decision_path_score" not in common_header
    assert "gafime_gpu_decision_path_membership" in cuda_rt_abi
    assert "gafime_gpu_decision_path_score" in cuda_rt_abi
    assert (
        "gafime_gpu_interaction_diagnostics"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    for launcher in (cuda_launcher, rocm_launcher, metal_launcher):
        assert "gafime_gpu_interaction_diagnostics" in launcher
        assert "InteractionDiagnostic" in launcher
    for kernels in (cuda_kernels, rocm_kernels, metal_shader):
        assert "interaction_diagnostics" in kernels
        assert "materialization" in kernels

    cpu_matrix = (ROOT / "crates" / "gafime-cpu" / "src" / "matrix.rs").read_text()
    cpu_diagnostics = (
        ROOT / "crates" / "gafime-cpu" / "src" / "diagnostics.rs"
    ).read_text()
    assert "minimums[col]" in cpu_matrix and "maximums[col]" in cpu_matrix
    assert "interaction_diagnostics" in cpu_diagnostics
    assert "prefix_product_is_f32_bounded" in cpu_diagnostics
    contract_text = (ROOT / "docs" / "contract.md").read_text()
    claude_text = (ROOT / "CLAUDE.md").read_text()
    agent_text = (ROOT / "AGENT.md").read_text()
    for policy_text in (contract_text, claude_text, agent_text):
        assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE=firsthit" in policy_text
        assert "non-overlapping" in policy_text
        assert "return unsupported" in policy_text
        assert "same selected shape and estimator" in policy_text
        assert "fixed equal-width MI" in policy_text
        assert "mi_bins` is an adaptive maximum" in policy_text
        assert "2,4,8,12,16,24,32,48,64,96" in policy_text
        assert "GAFIME_METAL_PARITY_TOLERANCE=0.00005" in policy_text
        assert "approved absolute fp32 release" in policy_text
        assert "ROCm managed storage requires both integrated placement" in policy_text
        assert "GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL" in policy_text
        assert "GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION" in policy_text
        assert "gafime_gpu_decision_path_release_device_state" in policy_text
        assert "compatibility mutex" in policy_text
    gpu_sys_standard = read_rust_crate_sources(
        "gafime-gpu-sys",
        include_test_modules=False,
        include_local_cmake_experiment=False,
    )
    gpu_sys = read_rust_crate_sources("gafime-gpu-sys", include_test_modules=False)
    gpu_sys_with_tests = read_rust_crate_sources("gafime-gpu-sys")
    assert "DEFAULT_METAL_PARITY_TOLERANCE: f32 = 5.0e-5" in gpu_sys_with_tests
    assert (
        "DEFAULT_METAL_FP32_CROSS_BACKEND_TOLERANCE: f32 = 2.0e-4"
        in gpu_sys_with_tests
    )
    assert "for rows in [160, 255, 256, 257]" in gpu_sys_with_tests
    assert "supports_decision_path_membership" not in gpu_sys_standard
    assert "supports_decision_path_score" not in gpu_sys_standard
    assert "supports_decision_path_membership" in gpu_sys
    assert "supports_decision_path_score" in gpu_sys
    assert "cuda_backend_with_local_cmake_experiment_for_test" in gpu_sys_with_tests
    assert (
        "cuda_decision_path_direct_score_groups_mixed_axes_when_rt_is_required"
        in gpu_sys_with_tests
    )
    assert (
        "cuda_decision_path_firsthit_score_partitioned_groups_match_cpu_when_rt_is_required"
        in gpu_sys_with_tests
    )
    assert (
        "cuda_decision_path_firsthit_score_rejects_overlap_without_sm_fallback"
        in gpu_sys_with_tests
    )
    assert (
        "cuda_decision_path_direct_score_recomputes_target_stats_with_cached_points"
        in gpu_sys_with_tests
    )
    assert (
        "cuda_decision_path_direct_score_refreshes_cached_scatter_map"
        in gpu_sys_with_tests
    )
    assert (
        "cuda_continuous_cached_target_stats_refresh_after_target_update"
        in gpu_sys_with_tests
    )
    assert (
        "rocm_continuous_cached_target_stats_refresh_after_target_update"
        in gpu_sys_with_tests
    )
    assert (
        "metal_device_topk_covers_split_directions_ties_and_large_k_when_available"
        in gpu_sys_with_tests
    )
    assert (
        "metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available"
        in gpu_sys_with_tests
    )
    assert (
        "metal_fp32_precision_metrics_match_core_fp32_on_high_dynamic_and_nonfinite_inputs_when_available"
        in gpu_sys_with_tests
    )
    for test_name in (
        "cuda_nonfinite_correlation_is_not_laundered_when_library_is_available",
        "rocm_nonfinite_correlation_is_not_laundered_when_library_is_available",
        "metal_nonfinite_correlation_is_not_laundered_when_library_is_available",
        "cuda_scaled_covariance_matches_cpu_across_dynamic_range_when_available",
        "rocm_scaled_covariance_matches_cpu_across_dynamic_range_when_available",
        "metal_scaled_covariance_matches_cpu_across_dynamic_range_when_available",
        "cuda_low_signal_mi_matches_cpu_when_library_is_available",
        "rocm_low_signal_mi_matches_cpu_when_library_is_available",
    ):
        assert test_name in gpu_sys_with_tests
    assert "covariance_requires_scaled_path" in covariance_policy
    assert "interaction_abs_exponent" in covariance_policy
    for launcher_text in (cuda_launcher, rocm_launcher, metal_launcher):
        assert "covariance_requires_scaled_path" in launcher_text
        assert "feature_abs_exponents" in launcher_text
        assert "target_abs_exponent" in launcher_text
    # CUDA ABI 1.1 owns one typed route engine: the accumulation type comes
    # from the selected profile rather than the historical one-mode-at-build
    # MI macro.  ROCm still carries the frozen ABI 1.0 macro only inside its
    # narrow compatibility primitive while its canonical route kernels are
    # independently typed.
    assert "__global__ void continuous_kernel" in cuda_kernels
    assert "__global__ void mutual_info_kernel" in cuda_kernels
    assert "struct PrecisionTraits<GAFIME_PRECISION_FP32>" in cuda_kernels
    assert "struct PrecisionTraits<GAFIME_PRECISION_MIXED>" in cuda_kernels
    assert "struct PrecisionTraits<GAFIME_PRECISION_FP64>" in cuda_kernels
    assert "__global__ void score_continuous_chunk_kernel_static" in rocm_kernels
    assert "GAFIME_GPU_MI_ACCUMULATION_FP64" in rocm_kernels
    assert "using MutualInfoAccumulator = double" in rocm_kernels
    assert "finalize_mutual_info_score" in rocm_kernels
    assert "scaled_covariance" in metal_shader
    assert "finalize_correlation" in cuda_kernels
    assert "device_nan" in cuda_kernels
    for source_path in (
        ROOT / "src" / "rocm" / "kernels.hip",
        ROOT / "src" / "metal" / "shader.metal",
    ):
        device_source = source_path.read_text()
        assert "finalize_correlation" in device_source
        assert "nonfinite_metric" in device_source
    metal_workflow = (
        ROOT / ".github" / "workflows" / "v1_contract_validation.yml"
    ).read_text()
    assert 'GAFIME_METAL_PARITY_TOLERANCE: "0.00005"' in metal_workflow
    assert "system_profiler SPHardwareDataType SPDisplaysDataType" in metal_workflow
    assert (
        "metal_device_topk_covers_split_directions_ties_and_large_k_when_available"
        in metal_workflow
    )
    assert (
        "metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available"
        in metal_workflow
    )
    assert (
        "metal_fp32_precision_metrics_match_core_fp32_on_high_dynamic_and_nonfinite_inputs_when_available"
        in metal_workflow
    )
    assert "uint32_t precision_profile;" in metal_launcher
    assert "uint precision_profile;" in metal_shader
    assert "sizeof(MetalLaunchInfo) == 24" in metal_launcher
    assert "matrix->precision_profile," in metal_launcher
    assert "const bool core_ordered_fp32_mean" in metal_shader
    assert "info.precision_profile == GAFIME_PRECISION_FP32" in metal_shader
    assert "info.rows <= static_cast<ulong>(4u * kMetalReduceWidth)" in metal_shader
    native_platform_workflow = (
        ROOT / ".github" / "workflows" / "native_platform_validation.yml"
    ).read_text()
    assert "run_legacy_payload_compatibility" in native_platform_workflow
    assert "GAFIME_METAL_PARITY_TOLERANCE=0.00005" in native_platform_workflow
    assert (
        "tests::metal::metal_continuous_metrics_match_cpu_on_high_dynamic_and_nonfinite_inputs_when_available"
        in native_platform_workflow
    )
    assert not (
        ROOT / "tests" / "release_measure" / "abi_02_legacy_gpu_payload_compatibility.py"
    ).exists()
    assert (
        "metal_nonfinite_correlation_is_not_laundered_when_library_is_available"
        in metal_workflow
    )
    assert (
        "metal_scaled_covariance_matches_cpu_across_dynamic_range_when_available"
        in metal_workflow
    )
    assert (
        "cuda_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available"
        in gpu_sys_with_tests
    )
    assert (
        "rocm_all_adaptive_mi_templates_match_cpu_for_arity_1_to_5_when_library_is_available"
        in gpu_sys_with_tests
    )
    assert (
        "rocm_adaptive_mi_96_matches_cpu_for_arity_1_to_5_when_library_is_available"
        in gpu_sys_with_tests
    )
    assert "GAFIME_REQUIRE_ROCM_WAVE64_MI" in gpu_sys_with_tests
    assert "configured CUDA payload failed to load" in gpu_sys_with_tests
    assert "configured ROCm payload failed to load" in gpu_sys_with_tests
    assert (
        "matrix->graph_metric_signature = metric_signature(protocol)"
        in (ROOT / "src" / "cuda" / "precision_launcher.cu").read_text()
    )
    assert (
        ROOT / "tests" / "release_measure" / "contract_04_adaptive_mi_quantization.py"
    ).exists()

    reduce_text = (
        ROOT / "crates" / "gafime-orchestrator" / "src" / "reduce" / "mod.rs"
    ).read_text()
    assert "CompactResultTablePlan" in reduce_text
    assert "planned_rows.min(rank.top_k as u64)" in reduce_text
    assert "10_000_000" in reduce_text
    assert (
        "result_plan_bounds_ten_million_candidate_metadata_without_records"
        in reduce_text
    )

    schedule_text = (
        ROOT / "crates" / "gafime-orchestrator" / "src" / "schedule" / "mod.rs"
    ).read_text()
    continuous_text = (
        ROOT / "crates" / "gafime-orchestrator" / "src" / "continuous.rs"
    ).read_text()
    assert "ContinuousSchedule" in schedule_text
    assert "CompactResultTablePlan::for_plan" in schedule_text
    assert "ContinuousSchedule::for_plan(&plan)" in continuous_text
    assert "self.schedule.result_table().capacity()" in continuous_text

    cpu_text = (ROOT / "crates" / "gafime-cpu" / "src" / "lib.rs").read_text()
    significance_text = (
        ROOT / "crates" / "gafime-cpu" / "src" / "significance.rs"
    ).read_text()
    assert "execute_ranked_continuous" in cpu_text
    assert "TopKSelector" in cpu_text
    assert "planned_rows.min(protocol.rank.top_k as u64)" in cpu_text
    assert "OwnedResultTable::new(2, 1, 2)" in cpu_text
    assert "select_adaptive_mi_bins_for_backend" in significance_text
    assert "params.backend_kind" in significance_text
    assert "significance_uses_fixed_width_mi" in significance_text
    assert (
        "significance_mi_bins_follow_the_observed_backend_ceiling" in significance_text
    )
    assert (
        "gpu_observations_force_fixed_width_mi_for_the_cpu_fallback"
        in significance_text
    )
    assert (
        "fixed_mi_significance_uses_the_observed_adaptive_template" in significance_text
    )


def check_pyo3_compact_report_and_cuda_surface() -> None:
    py_text = read_rust_crate_sources("gafime-py", include_local_cmake_experiment=False)
    python_report = (ROOT / "python" / "gafime" / "reporting" / "report.py").read_text()
    python_adapter = (ROOT / "python" / "gafime" / "v1_adapter.py").read_text()
    assert "table: SendOwnedResultTable" in py_text
    assert "struct SendOwnedResultTable" in py_text
    assert "PrecisionOwnedResultTable);" in py_text
    assert "records: Vec<PyContinuousRecord>" not in py_text
    assert "impl From<ContinuousReport> for PyContinuousReport" in py_text
    assert "table: SendOwnedResultTable::new(value.table)" in py_text
    assert "GpuBackend::cuda_from_env" in py_text
    assert '"auto" => Ok(resolve_auto_backend(device_id, precision))' in py_text
    assert "probe_gpu_candidate(GAFIME_BACKEND_CUDA" in py_text
    assert "GpuBackend::metal_from_env" in py_text
    assert "cpu_isa_rank(finite_dispatch_isa())" in py_text
    assert '"cuda" => Ok(GAFIME_BACKEND_CUDA)' in py_text
    assert '"gpu" => Err' in py_text
    assert "v1-cuda-cabi" in py_text
    assert "interaction_diagnostics_available" in py_text
    assert "fn interaction_diagnostic(" in py_text
    assert "fn interaction_diagnostics_batch(" in py_text
    assert "interaction_overflow_candidate_count" in py_text
    assert "interaction_overflow_max_rows" in py_text
    assert "interaction_diagnostics_cache" in py_text
    assert "cache.combo_indices == combo_indices" in py_text
    assert "self.interaction_diagnostics_cache = None" in py_text
    assert "interaction_overflow_rows" in python_report
    assert "precision_diagnostics_available" in python_report
    assert "_interaction_diagnostics_batch" in python_report
    assert "native diagnostic batch does not align" in python_report
    assert "self._precision = normalize_precision" in python_report
    assert 'if self._precision == "fp32":' in python_report
    assert 'return array("f", (ratio,))[0]' in python_report
    assert "precision=self._precision" in python_report
    assert "interaction_overflow_candidate_count" in python_adapter
    assert "interaction_overflow_max_rows" in python_adapter
    assert (
        'precision=getattr(native_report, "precision", config.precision)'
        in python_adapter
    )

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
    report = gafime.GafimeEngine(cfg).analyze(
        [[1.0, 0.0], [2.0, 1.0]], [1.0, 2.0], ["a", "b"]
    )
    assert fake.calls
    assert len(report.interactions) == 10_000_000
    top = report.interactions.top_k(3, metric_name="pearson")
    assert len(top) == 3
    assert [item.combo for item in top] == [(0,), (1,), (0,)]


def run_cargo(include_gpu: bool) -> None:
    env = os.environ.copy()
    if include_gpu:
        env["RUST_TEST_THREADS"] = "1"
        for key in (
            "GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP",
            "GAFIME_CUDA_DECISION_PATH_RT",
            "GAFIME_CUDA_DECISION_PATH_RT_SCORE",
        ):
            env.pop(key, None)
        required_values = {
            "GAFIME_CUDA_V1_LIB": env.get("GAFIME_CUDA_V1_LIB"),
            "GAFIME_CUDA_RT_V1_LIB": env.get("GAFIME_CUDA_RT_V1_LIB"),
            "GAFIME_ROCM_V1_LIB": env.get("GAFIME_ROCM_V1_LIB"),
            "GAFIME_CUDA_ABI_SMOKE": env.get(
                "GAFIME_CUDA_ABI_SMOKE", "/tmp/cuda_v1_abi_smoke"
            ),
            "GAFIME_CUDA_RT_ABI_SMOKE": env.get(
                "GAFIME_CUDA_RT_ABI_SMOKE", "/tmp/cuda_v1_abi_smoke_rt"
            ),
            "GAFIME_ROCM_ABI_SMOKE": env.get(
                "GAFIME_ROCM_ABI_SMOKE", "/tmp/rocm_v1_abi_smoke"
            ),
        }
        unset = [name for name, value in required_values.items() if not value]
        if unset:
            raise AssertionError(
                "missing configured GPU payload or ABI smoke paths: " + ", ".join(unset)
            )
        required = {
            name: Path(value).expanduser().resolve()
            for name, value in required_values.items()
            if value
        }
        missing = [
            f"{name}={path}" for name, path in required.items() if not path.is_file()
        ]
        if missing:
            raise AssertionError(
                "missing configured GPU payloads or ABI smokes: " + ", ".join(missing)
            )
        for name in (
            "GAFIME_CUDA_V1_LIB",
            "GAFIME_CUDA_RT_V1_LIB",
            "GAFIME_ROCM_V1_LIB",
        ):
            env[name] = str(required[name])
        inherited_library_path = env.get("LD_LIBRARY_PATH")

        def smoke_environment(payload: Path) -> dict[str, str]:
            smoke_env = env.copy()
            search_path = [str(payload.parent)]
            if inherited_library_path:
                search_path.append(inherited_library_path)
            smoke_env["LD_LIBRARY_PATH"] = os.pathsep.join(search_path)
            return smoke_env

        subprocess.run(
            [str(required["GAFIME_CUDA_ABI_SMOKE"])],
            cwd=ROOT,
            check=True,
            env=smoke_environment(required["GAFIME_CUDA_V1_LIB"]),
        )
        rt_smoke_env = smoke_environment(required["GAFIME_CUDA_RT_V1_LIB"])
        rt_smoke_env["GAFIME_CUDA_REQUIRE_RT_MEMBERSHIP"] = "1"
        subprocess.run(
            [str(required["GAFIME_CUDA_RT_ABI_SMOKE"])],
            cwd=ROOT,
            check=True,
            env=rt_smoke_env,
        )
        rocm_smoke_env = smoke_environment(required["GAFIME_ROCM_V1_LIB"])
        subprocess.run(
            [str(required["GAFIME_ROCM_ABI_SMOKE"])],
            cwd=ROOT,
            check=True,
            env=rocm_smoke_env,
        )
        env["GAFIME_CUDA_V1_LIB"] = str(required["GAFIME_CUDA_RT_V1_LIB"])
        env["GAFIME_REQUIRE_CURRENT_DEVICE_RANKING"] = "1"
        cargo_library_path = [
            str(required["GAFIME_CUDA_RT_V1_LIB"].parent),
            str(required["GAFIME_ROCM_V1_LIB"].parent),
        ]
        if inherited_library_path:
            cargo_library_path.append(inherited_library_path)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(cargo_library_path)
    subprocess.run(
        ["cargo", "test", "--workspace", "--", "--test-threads=1"],
        cwd=ROOT,
        check=True,
        env=env,
    )


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
    check_compatibility_scenario_metadata()
    if not args.skip_cargo:
        run_cargo(args.include_gpu)
    print("v1 architecture gate passed")


if __name__ == "__main__":
    main()
