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
    crate_name: str, *, include_test_modules: bool = True
) -> str:
    source_root = ROOT / "crates" / crate_name / "src"
    paths = sorted(source_root.rglob("*.rs"))
    if not include_test_modules:
        paths = [
            path
            for path in paths
            if "tests" not in path.relative_to(source_root).parts
        ]
    return "\n".join(
        path.read_text() for path in paths
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
    assert '#[target_feature(enable = "avx2")]' in covariance_text
    assert '#[target_feature(enable = "sse4.2")]' in covariance_text
    assert '#[target_feature(enable = "avx2")]' in histogram_text
    finite_body = covariance_text.split("fn pearson_sums_finite", 1)[1].split(
        "fn all_pairs_finite", 1
    )[0]
    assert "pearson_sums_avx2" in finite_body
    assert "pearson_sums_sse42" in finite_body
    assert "pearson_sums_neon" in finite_body

    matrix_text = (ROOT / "crates" / "gafime-cpu" / "src" / "matrix.rs").read_text()
    assert "columns: Vec<f32>" in matrix_text
    assert "transpose_row_major" in matrix_text
    assert "self.columns[col * self.rows as usize + row]" in matrix_text

    kernels_text = (
        ROOT / "crates" / "gafime-cpu" / "src" / "kernels" / "mod.rs"
    ).read_text()
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
    cuda_api = (cuda_root / "cuda_api.hpp").read_text()
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
    stage_gpu_payload = (
        ROOT / ".github" / "scripts" / "stage_gpu_payload.py"
    ).read_text()
    common_header = (common_root / "gafime_gpu_abi.hpp").read_text()
    python_config = (ROOT / "python" / "gafime" / "config.py").read_text()
    python_precision = (ROOT / "python" / "gafime" / "_precision.py").read_text()
    python_capabilities = (ROOT / "python" / "gafime" / "capabilities.py").read_text()
    python_adapter = (ROOT / "python" / "gafime" / "v1_adapter.py").read_text()
    rust_runtime = (ROOT / "crates" / "gafime-py" / "src" / "runtime.rs").read_text()
    precision_doc = (ROOT / "docs" / "precision-contract.md").read_text()
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
        (cuda_abi_smoke, ("protocol", "stable_protocol", "mi_protocol", "mixed_protocol")),
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
    assert (
        "GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT 0u" in common_header
    )
    for name, launcher_text in (
        ("cuda", cuda_launcher),
        ("rocm", rocm_launcher),
        ("metal", metal_launcher),
    ):
        assert "GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL" in launcher_text, name
        assert "GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL" in launcher_text, name
        assert "GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION" in launcher_text, name
        assert "descriptors_resident" in launcher_text, name
        assert "invalidate_protocol_descriptor_cache" in launcher_text, name
        assert "GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT" in launcher_text, name
        assert "descriptor_generation != 0" in launcher_text, name
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
    assert "CudaDeviceBufferReservation" in cuda_launcher
    assert "HipDeviceBufferReservation" in rocm_launcher
    for launcher_text, reservation_name in (
        (cuda_launcher, "CudaDeviceBufferReservation"),
        (rocm_launcher, "HipDeviceBufferReservation"),
    ):
        assert "struct DescriptorBufferUpdateTransition" in launcher_text
        assert (
            "DescriptorBufferUpdateTransition{true, false, false, false}"
            in launcher_text
        )
        assert (
            "DescriptorBufferUpdateTransition{true, true, true, false}"
            in launcher_text
        )
        first_reservation = launcher_text.index(
            f"{reservation_name}<uint32_t> combo_reservation"
        )
        second_reservation = launcher_text.index(
            f"{reservation_name}<uint32_t> metric_id_reservation",
            first_reservation,
        )
        cache_invalidation = launcher_text.index(
            "invalidate_protocol_descriptor_cache(matrix)", second_reservation
        )
        assert first_reservation < second_reservation < cache_invalidation
        assert "replaces_live_buffer()" in launcher_text[second_reservation:]
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
    assert "protocol.reserved[DESCRIPTOR_GENERATION_RESERVED_SLOT]" in continuous_execution
    assert "descriptor_generation: next_descriptor_generation()" in continuous_execution
    gpu_sys = read_rust_crate_sources(
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
    assert "legacy_cuda_decision_path_lock" in gpu_sys
    assert "negotiated.flags &= !GAFIME_LAUNCH_FLAG_IMMUTABLE_PROTOCOL" in gpu_sys
    assert (
        "negotiated.reserved[GAFIME_LAUNCH_PROTOCOL_DESCRIPTOR_GENERATION_SLOT] = 0"
        in gpu_sys
    )
    assert "cargo +1.89.0 test --workspace --quiet" in contract_workflow
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
    assert "gafime_gpu_decision_path_release_device_state" in common_header
    assert "GAFIME_CUDA_DECISION_PATH_RT_SCORE" in cuda_abi_smoke
    assert "disabled_firsthit_status" in cuda_abi_smoke
    assert "cudaSetDevice(1) before RT calls" in cuda_rt_concurrency
    assert "cudaGetDevice after RT cleanup" in cuda_rt_concurrency
    assert "gafime_gpu_decision_path_release_device_state" in cuda_rt_state_policy

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

    assert "<<<" in cuda_launcher, "CUDA launcher owns <<<>>> launch calls"
    assert "<<<" in cuda_rt_launcher, "CUDA RT launcher owns RT <<<>>> launch calls"
    assert "<<<" not in cuda_kernels, "CUDA kernels file must not own launches"
    assert "<<<" not in cuda_rt_kernels, "CUDA RT kernels file must not own launches"
    assert "hipLaunchKernelGGL" in rocm_launcher, "ROCm launcher owns HIP launch calls"
    assert "hipLaunchKernelGGL" not in rocm_kernels and "<<<" not in rocm_kernels

    for name, device_text in (("cuda", cuda_kernels), ("rocm", rocm_kernels)):
        assert "__global__ void score_continuous_chunk_kernel" in device_text, name
        assert "__global__ void target_stats_kernel" in device_text, name
        assert "__global__ void unary_feature_stats_kernel" in device_text, name
        assert (
            "__global__ void score_continuous_unary_all_finite_chunk_kernel"
            in device_text
        ), name
        assert "target_stats->finite" in device_text, name
        assert "feature_stats[col].sxx" in device_text, name
        assert "__global__ void score_spearman_chunk_kernel" in device_text, name
        assert "__global__ void score_mutual_info_chunk_kernel" in device_text, name
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
        assert "static_cast<uint64_t>(col) * rows + row" in device_text, (
            f"{name} kernels must read feature-major resident features"
        )
        assert (
            "interaction_value(features, column_means, row, n_samples" in device_text
        ), f"{name} kernels must pass rows as the feature-major stride"
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
    assert "params.membership_words = program.membership_words_device;" not in planned_body
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
    assert "gafime_gpu_decision_path_membership" not in cuda_rt_launcher
    assert "gafime_gpu_decision_path_score" not in cuda_rt_launcher
    assert "props.major == 8 && props.minor >= 9" in cuda_launcher
    assert "gafime_gpu_permutation_memory_peak" in cuda_launcher
    assert "gafime_gpu_permutation_pvalues" in cuda_launcher
    assert "gafime_gpu_decision_path_membership" in cuda_launcher
    assert "gafime_gpu_decision_path_score" in cuda_launcher
    assert "execute_decision_path_membership" in cuda_launcher
    assert "execute_decision_path_score" in cuda_launcher
    assert "has_continuous_covariance_metric" in cuda_launcher
    assert "if (has_continuous_covariance_metric(protocol))" in cuda_launcher
    assert "has_continuous_covariance_metric" in rocm_launcher
    assert "if (has_continuous_covariance_metric(protocol))" in rocm_launcher
    for launcher_text in (cuda_launcher, rocm_launcher):
        assert "hint == 12" in launcher_text
        assert "hint == 24" in launcher_text
        assert "hint == 48" in launcher_text
        assert "case 12:" in launcher_text
        assert "case 24:" in launcher_text
        assert "case 48:" in launcher_text
    for device_text in (cuda_kernels, rocm_kernels):
        assert "score_continuous_chunk_kernel_static" in device_text
        assert "score_mutual_info_chunk_kernel_static" in device_text
        assert "score_spearman_chunk_kernel_static" in device_text
        assert "select_topk_partials_kernel_static" in device_text
        assert "merge_topk_partials_kernel_static" in device_text
        assert "copy_selected_metric_rows_kernel" in device_text
    for device_text in (cuda_kernels, rocm_kernels, metal_shader):
        assert "previous_score" in device_text
        assert "previous_index" in device_text
        assert "already_selected" not in device_text
    assert "GAFIME_CUDA_FORCEINLINE" in cuda_kernels
    assert "GAFIME_HIP_FORCEINLINE" in rocm_kernels
    assert "hint == 16 || hint == 24 || hint == 32 || hint == 48" in metal_launcher
    assert "std::vector<double> sums(cols, 0.0)" in metal_launcher
    assert "sums[col] += static_cast<double>(features[base + col])" in metal_launcher
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
    assert "require-topk-split" in static_kernel_report
    assert "require-no-spills" in static_kernel_report
    assert "__ockl_wfred_min_u32" in rocm_kernels
    assert "__ockl_wfred_max_u32" in rocm_kernels
    assert "__ockl_wfred_add_u32" in rocm_kernels
    assert (
        "reduce_float0[threadIdx.x] += reduce_float0[threadIdx.x + stride]"
        in rocm_kernels
    )
    for launcher_text in (cuda_launcher, rocm_launcher, metal_launcher):
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

    for name, header_text in (("cuda", cuda_header), ("rocm", rocm_header)):
        assert "namespace kernel" in header_text, name
        assert "TargetStatsDevice" in header_text, name
        assert "UnaryFeatureStatsDevice" in header_text, name
        assert "launch_target_stats" in header_text, name
        assert "launch_unary_feature_stats" in header_text, name
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
    assert "bool content_valid;" in metal_launcher
    assert "matrix->content_valid = false;" in metal_launcher
    assert "if (!matrix->content_valid)" in metal_launcher

    for name, launcher_text, content_marker in (
        ("cuda", cuda_launcher, "require_valid_matrix_content(matrix)"),
        ("rocm", rocm_launcher, "require_valid_matrix_content(matrix)"),
        ("metal", metal_launcher, "if (!matrix->content_valid)"),
    ):
        execute_body = launcher_text.split(
            "GAFIME_GPU_API int gafime_gpu_execute", 1
        )[1]
        assert execute_body.index("validate_protocol") < execute_body.index(
            content_marker
        ), f"{name} checks content before protocol ABI"
        assert execute_body.index("validate_result_table") < execute_body.index(
            content_marker
        ), f"{name} checks content before result ABI"

    cuda_membership_body = cuda_launcher.split(
        "GAFIME_GPU_API int gafime_gpu_decision_path_membership", 1
    )[1]
    assert cuda_membership_body.index("paths->abi_version") < cuda_membership_body.index(
        "require_valid_matrix_content(matrix)"
    )
    assert "GAFIME_MAX_DECISION_PATH_COUNT" in cuda_rt_launcher

    assert "defined(GAFIME_BUILDING_DLL)" in cuda_api
    assert "#define GAFIME_GPU_BUILDING_DLL" in cuda_api
    assert "GAFIME_GPU_BUILDING_DLL" in cuda_cmake
    assert "-DGAFIME_BUILDING_DLL" not in stage_gpu_payload
    assert stage_gpu_payload.count("-DGAFIME_GPU_BUILDING_DLL") == 2
    assert stage_gpu_payload.count("-DGAFIME_GPU_MI_ACCUMULATION_FP64=0") == 2
    assert '"covariance_policy.hpp"' in stage_gpu_payload

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
        "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT",
        "GAFIME_GPU_ARCH_NVIDIA_ADA",
        "GAFIME_GPU_ARCH_AMD_CDNA",
        "GAFIME_GPU_ARCH_APPLE",
        "GAFIME_DECISION_PATH_FLAG_REQUIRE_RT",
    ):
        assert marker in common_header, marker

    assert "cuda_arch_class" in cuda_launcher
    assert "cuda_device_flags" in cuda_launcher
    assert "GAFIME_GPU_DEVICE_FLAG_OPTIX_RT" in cuda_launcher
    assert "defined(GAFIME_CUDA_ENABLE_OPTIX_RT)" in cuda_launcher
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
    assert "GAFIME_CUDA_EXPECT_NO_RT" in cuda_abi_smoke
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

    assert "kernels.cu" in cuda_cmake and "launcher.cu" in cuda_cmake
    assert "rt_kernels.cu" in cuda_cmake and "rt_launcher.cu" in cuda_cmake
    assert "GAFIME_CUDA_ENABLE_OPTIX_RT" in cuda_cmake
    assert "GAFIME_CUDA_RT_BUILD_MODE" in cuda_cmake
    assert "^(off|on|both)$" in cuda_cmake
    assert 'set(GAFIME_CUDA_MI_ACCUMULATION_MODE "fast" CACHE STRING' in cuda_cmake
    assert "^(fast|fp64)$" in cuda_cmake
    assert "GAFIME_GPU_MI_ACCUMULATION_FP64" in cuda_cmake
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
    assert 'set(GAFIME_HIP_MI_ACCUMULATION_MODE "fast" CACHE STRING' in rocm_cmake
    assert "^(fast|fp64)$" in rocm_cmake
    assert "GAFIME_GPU_MI_ACCUMULATION_FP64" in rocm_cmake
    assert "-DGAFIME_CUDA_MI_ACCUMULATION_MODE=fp64" in contract_workflow
    assert "GAFIME_EXPECT_MI_ACCUMULATION_FP64" in cuda_cmake
    assert "GAFIME_EXPECT_MI_ACCUMULATION_FP64" in rocm_cmake
    assert "ctest --test-dir build/ci-cuda-mi-fp64" in contract_workflow

    assert 'storage_dtype: str = "float32"' in python_config
    assert 'compute_policy: str = "stable"' in python_config
    assert "SUPPORTED_STORAGE_DTYPES" in python_precision
    assert "unsupported_precision_reason" in python_adapter
    assert "precision_contract: CapabilityValue" in python_capabilities
    assert 'get_string(config, "storage_dtype", "float32")' in rust_runtime
    assert 'get_string(config, "compute_policy", "stable")' in rust_runtime
    assert 'runtime.set_item("precision", precision)' in rust_runtime
    assert "GAFIME_DTYPE_F64 = 2" in precision_doc
    assert "must not label a partial wider" in precision_doc


def check_native_abi_and_reduce_scale_structure() -> None:
    types_text = (ROOT / "crates" / "gafime-types" / "src" / "lib.rs").read_text()
    covariance_policy = (
        ROOT / "src" / "common" / "covariance_policy.hpp"
    ).read_text()
    cuda_launcher = (ROOT / "src" / "cuda" / "launcher.cu").read_text()
    cuda_kernels = (ROOT / "src" / "cuda" / "kernels.cu").read_text()
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
    assert "GafimeDecisionPathTerm" in types_text
    assert "GafimeDecisionPathBatch" in types_text
    assert "GafimeDecisionPathScoreBatch" in types_text
    assert (
        "gafime_gpu_permutation_memory_peak"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    assert (
        "gafime_gpu_permutation_pvalues"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    assert (
        "gafime_gpu_decision_path_membership"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
    assert (
        "gafime_gpu_decision_path_score"
        in (ROOT / "src" / "common" / "gafime_gpu_abi.hpp").read_text()
    )
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
    gpu_sys = read_rust_crate_sources(
        "gafime-gpu-sys", include_test_modules=False
    )
    gpu_sys_with_tests = read_rust_crate_sources("gafime-gpu-sys")
    assert "DEFAULT_METAL_PARITY_TOLERANCE: f32 = 5.0e-5" in gpu_sys_with_tests
    assert "supports_decision_path_membership" in gpu_sys
    assert "supports_decision_path_score" in gpu_sys
    assert "cuda_backend_with_optix_rt_for_test" in gpu_sys_with_tests
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
    for device_text in (cuda_kernels, rocm_kernels):
        assert "score_continuous_scaled_chunk_kernel" in device_text
        assert "GAFIME_GPU_MI_ACCUMULATION_FP64" in device_text
        assert "using MutualInfoAccumulator = double" in device_text
        assert "finalize_mutual_info_score" in device_text
    assert "scaled_covariance" in metal_shader
    for source_path in (
        ROOT / "src" / "cuda" / "kernels.cu",
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
        "graph_metric_signature = compute_metric_signature(protocol)"
        in (ROOT / "src" / "cuda" / "launcher.cu").read_text()
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
    assert "significance_mi_bins_follow_the_observed_backend_ceiling" in significance_text
    assert "gpu_observations_force_fixed_width_mi_for_the_cpu_fallback" in significance_text
    assert (
        "fixed_mi_significance_uses_the_observed_adaptive_template" in significance_text
    )


def check_pyo3_compact_report_and_cuda_surface() -> None:
    py_text = read_rust_crate_sources("gafime-py")
    assert "table: OwnedResultTable" in py_text
    assert "struct SendOwnedResultTable" in py_text
    assert "OwnedResultTable);" in py_text
    assert "records: Vec<PyContinuousRecord>" not in py_text
    assert "impl From<ContinuousReport> for PyContinuousReport" in py_text
    assert "table: SendOwnedResultTable(value.table)" in py_text
    assert "GpuBackend::cuda_from_env" in py_text
    assert '"auto" => Ok(resolve_auto_backend(device_id))' in py_text
    assert "probe_gpu_candidate(GAFIME_BACKEND_CUDA" in py_text
    assert "GpuBackend::metal_from_env" in py_text
    assert "cpu_isa_rank(finite_dispatch_isa())" in py_text
    assert '"cuda" => Ok(GAFIME_BACKEND_CUDA)' in py_text
    assert '"gpu" => Err' in py_text
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
            "GAFIME_CUDA_EXPECT_NO_RT",
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

        no_rt_smoke_env = smoke_environment(required["GAFIME_CUDA_V1_LIB"])
        no_rt_smoke_env["GAFIME_CUDA_EXPECT_NO_RT"] = "1"
        subprocess.run(
            [str(required["GAFIME_CUDA_ABI_SMOKE"])],
            cwd=ROOT,
            check=True,
            env=no_rt_smoke_env,
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
