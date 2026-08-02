#!/usr/bin/env python3
"""Clean installed-payload discovery, separation, and ABI-export smoke."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


REQUIRED_GPU_ABI_SYMBOLS = (
    "gafime_gpu_device_info",
    "gafime_gpu_graph_capability",
    "gafime_gpu_matrix_alloc",
    "gafime_gpu_matrix_upload",
    "gafime_gpu_matrix_update_target",
    "gafime_gpu_matrix_free",
    "gafime_gpu_execute",
)

REQUIRED_PRECISION_ABI_SYMBOLS = (
    "gafime_gpu_precision_capabilities",
    "gafime_gpu_matrix_alloc_v2",
    "gafime_gpu_matrix_upload_f32_v2",
    "gafime_gpu_matrix_upload_f64_v2",
    "gafime_gpu_matrix_update_target_f32_v2",
    "gafime_gpu_matrix_update_target_f64_v2",
    "gafime_gpu_execute_f32_v2",
    "gafime_gpu_execute_f64_v2",
    "gafime_gpu_execution_memory_peak_v2",
)

OPTIONAL_PRECISION_PERMUTATION_ABI_SYMBOLS = (
    "gafime_gpu_permutation_memory_peak_v2",
    "gafime_gpu_permutation_pvalues_f32_v2",
    "gafime_gpu_permutation_pvalues_f64_v2",
)

PRECISION_ABI_VERSION = (1 << 16) | 1
PRECISION_PROFILE_MASKS = {"fp32": 0x1, "mixed": 0x2, "fp64": 0x4}
DTYPE_MASK_F32 = 0x1
DTYPE_MASK_F64 = 0x2
BACKEND_KINDS = {"cuda": 2, "rocm": 3, "metal": 4}
EXPECTED_PROFILES = {
    "cuda": ("fp32", "mixed", "fp64"),
    "rocm": ("fp32", "mixed", "fp64"),
    "metal": ("fp32",),
}


class GafimePrecisionCapabilities(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("backend_kind", ctypes.c_uint32),
        ("profile_mask", ctypes.c_uint32),
        ("storage_dtype_mask", ctypes.c_uint32),
        ("result_dtype_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    )


PAYLOADS = {
    "cuda": {
        "distribution": "gafime-cuda",
        "package": "gafime_cuda",
        "env": "GAFIME_CUDA_V1_LIB",
    },
    "rocm": {
        "distribution": "gafime-rocm",
        "package": "gafime_rocm",
        "env": "GAFIME_ROCM_V1_LIB",
    },
    "metal": {
        "distribution": "gafime",
        "package": "gafime",
        "env": "GAFIME_METAL_V1_LIB",
    },
}

CUDA_BUILD_POLICY = {
    "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
    "cuda_tuning_policy": "runtime-device-class",
    "cuda_tuning_sm": None,
    "cuda_runtime": "system",
    "cuda_runtime_libraries": {
        "linux": "libcudart.so.13",
        "windows": "nvcudart_hybrid64.dll",
    },
    "optix_rt": "off",
    "precision_abi_version": "1.1",
    "precision_profiles": ["fp32", "mixed", "fp64"],
    "per_architecture_tuning": False,
    "rt_sources_included": False,
    "runtime_architecture_dispatch": True,
}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _remove_checkout_paths(source_root: Path) -> None:
    clean_path: list[str] = []
    for entry in sys.path:
        try:
            resolved = (Path(entry) if entry else Path.cwd()).resolve()
        except OSError:
            clean_path.append(entry)
            continue
        if not _is_within(resolved, source_root):
            clean_path.append(entry)
    sys.path[:] = clean_path


def _assert_installed(module: object, source_root: Path, label: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise AssertionError(f"{label} has no import path")
    path = Path(raw_path).resolve()
    if _is_within(path, source_root):
        raise AssertionError(f"{label} imported from checkout: {path}")
    if not path.is_file():
        raise AssertionError(f"{label} import path does not exist: {path}")
    return path


def _distribution_files(distribution: importlib.metadata.Distribution) -> set[str]:
    if distribution.files is None:
        raise AssertionError(
            f"{distribution.metadata['Name']} has no installed file manifest"
        )
    return {str(path).replace("\\", "/") for path in distribution.files}


def _assert_distribution_license(distribution: importlib.metadata.Distribution) -> None:
    if distribution.metadata.get("License-Expression") != "Apache-2.0":
        raise AssertionError(
            f"{distribution.metadata['Name']} does not declare Apache-2.0 license metadata"
        )
    license_files = distribution.metadata.get_all("License-File") or []
    if not any(Path(value).name == "LICENSE" for value in license_files):
        raise AssertionError(
            f"{distribution.metadata['Name']} does not declare LICENSE"
        )
    files = _distribution_files(distribution)
    if not any(Path(value).name == "LICENSE" for value in files):
        raise AssertionError(
            f"{distribution.metadata['Name']} does not install LICENSE"
        )


def _assert_cuda_build_policy(payload_module: object) -> None:
    module_path = Path(str(payload_module.__file__)).resolve().parent
    policy_path = module_path / "build_policy.json"
    if not policy_path.is_file():
        raise AssertionError(
            f"installed CUDA payload has no build policy: {policy_path}"
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy != CUDA_BUILD_POLICY:
        raise AssertionError(
            f"installed CUDA build policy {policy!r} != {CUDA_BUILD_POLICY!r}"
        )


def _assert_cuda_distribution_surface(library: Path) -> None:
    loaded = ctypes.CDLL(str(library))
    for symbol in (
        "gafime_gpu_decision_path_membership",
        "gafime_gpu_decision_path_score",
        "gafime_gpu_decision_path_release_device_state",
    ):
        try:
            getattr(loaded, symbol)
        except AttributeError:
            continue
        raise AssertionError(
            f"standard CUDA distribution unexpectedly exports local RT symbol {symbol}"
        )


def _assert_rocm_build_policy(
    payload_module: object, source_root: Path
) -> dict[str, object]:
    module_path = Path(str(payload_module.__file__)).resolve().parent
    policy_path = module_path / "build_policy.json"
    if not policy_path.is_file():
        raise AssertionError(
            f"installed ROCm payload has no build policy: {policy_path}"
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    wheel_policy = policy.get("wheel_policy")
    if wheel_policy != "system":
        raise AssertionError(
            f"installed ROCm wheel policy is invalid: {wheel_policy!r}"
        )
    expected_path = (
        source_root / ".github" / "scripts" / "rocm_7_2_3_system_policy.json"
    )
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if policy != expected:
        raise AssertionError(
            f"installed ROCm build policy differs from {expected_path}"
        )
    return policy


def _assert_linux_rocm_runtime(
    library: Path, package_path: Path, policy: dict[str, object]
) -> None:
    if sys.platform != "linux":
        return
    if policy.get("wheel_policy") != "system":
        raise AssertionError("distributed ROCm payload must use the system policy")
    private_dir = package_path.parent / f"{package_path.name}.libs"
    if private_dir.exists():
        raise AssertionError(
            f"system ROCm policy unexpectedly bundled userspace: {private_dir}"
        )
    dynamic = subprocess.run(
        ["readelf", "-d", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "(RPATH)" in dynamic.stdout or "(RUNPATH)" in dynamic.stdout:
        raise AssertionError(
            "system ROCm payload must use the system dynamic loader without "
            "an embedded RPATH or RUNPATH"
        )
    result = subprocess.run(
        ["ldd", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    if "not found" in result.stdout or "not found" in result.stderr:
        raise AssertionError(
            "installed system ROCm prerequisite is incomplete:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    runtime_lines = [
        line for line in result.stdout.splitlines() if "libamdhip64.so.7" in line
    ]
    if len(runtime_lines) != 1:
        raise AssertionError(
            "system ROCm payload did not resolve exactly one libamdhip64.so.7: "
            f"{runtime_lines}"
        )
    if "gafime_rocm" in runtime_lines[0]:
        raise AssertionError(
            f"system ROCm payload resolved a wheel-private runtime: {runtime_lines[0]}"
        )
    ctypes.CDLL(str(library), mode=getattr(os, "RTLD_LOCAL", 0))


def _assert_payload_separation(backend: str, payload: dict[str, str]) -> None:
    base = importlib.metadata.distribution("gafime")
    base_files = _distribution_files(base)
    vendor_entries = (
        "gafime_cuda/",
        "gafime_cuda_rt/",
        "gafime_rocm/",
        "gafime_rocm_bundled/",
        "libgafime_cuda",
        "gafime_cuda.dll",
        "gafime_cuda_rt.dll",
        "libgafime_rocm",
        "gafime_rocm.dll",
    )
    leaked = sorted(
        path for path in base_files if any(entry in path for entry in vendor_entries)
    )
    if leaked:
        raise AssertionError(
            f"base gafime distribution contains vendor payload files: {leaked}"
        )

    if backend == "metal":
        expected = {
            "gafime/_metal/libgafime_metal_v1.dylib",
            "gafime/_metal/gafime_metal_v1.metallib",
        }
        missing = sorted(expected - base_files)
        if missing:
            raise AssertionError(
                f"base macOS wheel is missing bundled Metal artifacts: {missing}"
            )
        return

    distribution = importlib.metadata.distribution(payload["distribution"])
    _assert_distribution_license(distribution)
    payload_files = _distribution_files(distribution)
    package_prefix = f"{payload['package']}/"
    if not any(path.startswith(package_prefix) for path in payload_files):
        raise AssertionError(
            f"{payload['distribution']} does not contain its {payload['package']} package"
        )
    vendor_payloads = (PAYLOADS["cuda"], PAYLOADS["rocm"])
    other_packages = {
        f"{candidate['package']}/"
        for candidate in vendor_payloads
        if candidate["package"] != payload["package"]
    }
    other_native_names = {
        native_name
        for candidate in vendor_payloads
        if candidate["package"] != payload["package"]
        for native_name in (
            f"lib{candidate['package']}.so",
            f"{candidate['package']}.dll",
            f"{candidate['package']}.so",
            f"{candidate['package']}.pyd",
        )
    }
    leaked = sorted(
        path
        for path in payload_files
        if any(path.startswith(package) for package in other_packages)
        or Path(path).name in other_native_names
    )
    if leaked:
        raise AssertionError(
            f"{payload['distribution']} contains another payload variant: {leaked}"
        )


def _assert_exported_symbols(library: Path, backend: str) -> None:
    if sys.platform == "win32":
        command = ["dumpbin", "/exports", str(library)]
    elif sys.platform == "darwin":
        command = ["nm", "-gU", str(library)]
    else:
        command = ["nm", "-D", "--defined-only", str(library)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AssertionError(
            f"required ABI inspection tool is unavailable: {command[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"unable to inspect payload ABI exports with {' '.join(command)}:\n{exc.stderr}"
        ) from exc
    missing = [
        symbol
        for symbol in (*REQUIRED_GPU_ABI_SYMBOLS, *REQUIRED_PRECISION_ABI_SYMBOLS)
        if symbol not in result.stdout
    ]
    if missing:
        raise AssertionError(
            f"payload library is missing required GPU ABI exports: {missing}"
        )
    exported_permutation = tuple(
        symbol
        for symbol in OPTIONAL_PRECISION_PERMUTATION_ABI_SYMBOLS
        if symbol in result.stdout
    )
    expected_permutation = (
        OPTIONAL_PRECISION_PERMUTATION_ABI_SYMBOLS if backend == "cuda" else ()
    )
    if exported_permutation != expected_permutation:
        raise AssertionError(
            f"{backend} native precision permutation exports {exported_permutation!r} "
            f"!= {expected_permutation!r}; ROCm/Metal use Rust orchestration"
        )


def _assert_precision_capability_abi(library: Path, backend: str) -> None:
    """Physically query the selected device without allocating a matrix."""

    loaded = ctypes.CDLL(str(library))
    query = loaded.gafime_gpu_precision_capabilities
    query.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(GafimePrecisionCapabilities),
    ]
    query.restype = ctypes.c_int
    capabilities = GafimePrecisionCapabilities()
    status = int(query(0, ctypes.byref(capabilities)))
    if status != 0:
        raise AssertionError(
            f"{backend} precision capability ABI failed on device 0 with status {status}"
        )
    expected_profiles = EXPECTED_PROFILES[backend]
    expected_profile_mask = sum(
        PRECISION_PROFILE_MASKS[profile] for profile in expected_profiles
    )
    expected_dtype_mask = (
        DTYPE_MASK_F32 if backend == "metal" else DTYPE_MASK_F32 | DTYPE_MASK_F64
    )
    if capabilities.abi_version != PRECISION_ABI_VERSION:
        raise AssertionError(
            f"{backend} precision ABI {capabilities.abi_version:#x} != "
            f"{PRECISION_ABI_VERSION:#x}"
        )
    if capabilities.backend_kind != BACKEND_KINDS[backend]:
        raise AssertionError(
            f"{backend} precision ABI reported backend kind {capabilities.backend_kind}"
        )
    if capabilities.profile_mask != expected_profile_mask:
        raise AssertionError(
            f"{backend} profile mask {capabilities.profile_mask:#x} != "
            f"{expected_profile_mask:#x}"
        )
    if capabilities.storage_dtype_mask != expected_dtype_mask:
        raise AssertionError(
            f"{backend} storage dtype mask "
            f"{capabilities.storage_dtype_mask:#x} != {expected_dtype_mask:#x}"
        )
    if capabilities.result_dtype_mask != expected_dtype_mask:
        raise AssertionError(
            f"{backend} result dtype mask "
            f"{capabilities.result_dtype_mask:#x} != {expected_dtype_mask:#x}"
        )


def _assert_public_precision_capabilities(
    gafime: object, backend: str, *, probe: bool
) -> None:
    expected_profiles = EXPECTED_PROFILES[backend]
    for profile in ("fp32", "mixed", "fp64"):
        expected_supported = profile in expected_profiles
        capabilities = gafime.backend_capabilities(
            backend,
            probe=probe and expected_supported,
            precision=profile,
        )
        precision = capabilities.precision_contract.value
        if tuple(precision["supported_profiles"]) != expected_profiles:
            raise AssertionError(
                f"{backend} public profiles {precision['supported_profiles']!r} != "
                f"{expected_profiles!r}"
            )
        if bool(precision["request_supported"]) is not expected_supported:
            raise AssertionError(
                f"{backend} precision={profile!r} support was reported as "
                f"{precision['request_supported']!r}"
            )
        if expected_supported and probe and precision["effective"] != profile:
            raise AssertionError(
                f"{backend} precision={profile!r} was not reported effective"
            )
        if not expected_supported:
            reason = str(precision.get("rejection_reason") or "")
            if "Metal supports precision='fp32' only" not in reason:
                raise AssertionError(
                    f"{backend} precision={profile!r} rejection was not actionable: "
                    f"{reason!r}"
                )
            if probe:
                try:
                    gafime.backend_capabilities(
                        backend,
                        probe=True,
                        precision=profile,
                    )
                except ValueError as exc:
                    if "Metal supports precision='fp32' only" not in str(exc):
                        raise AssertionError(
                            f"{backend} precision={profile!r} probe rejection was "
                            f"not actionable: {exc}"
                        ) from exc
                else:
                    raise AssertionError(
                        f"{backend} precision={profile!r} probe did not fail closed"
                    )


def _assert_package_helpers(
    payload_module: object, package_path: Path, library: Path
) -> None:
    package_dir = getattr(payload_module, "package_dir", None)
    library_candidates = getattr(payload_module, "library_candidates", None)
    if not callable(package_dir) or not callable(library_candidates):
        raise AssertionError(
            "installed payload package must expose package_dir() and "
            "library_candidates()"
        )
    if Path(package_dir()).resolve() != package_path:
        raise AssertionError("installed payload package_dir() returned another package")
    candidates = [Path(path).resolve() for path in library_candidates()]
    if library not in candidates:
        raise AssertionError(
            f"installed payload library_candidates() omitted selected library: {library}"
        )
    if any(path.parent != package_path for path in candidates):
        raise AssertionError(
            "installed payload library_candidates() escaped its package directory"
        )


def _exercise_metal_public_api(gafime: object) -> None:
    config = gafime.EngineConfig(
        backend="metal",
        precision="fp32",
        metric_names=("pearson", "spearman", "mutual_info", "r2"),
        permutation_tests=0,
        num_repeats=1,
        budget=gafime.ComputeBudget(max_comb_size=1, max_combinations_per_k=8),
    )
    report = gafime.GafimeEngine(config).analyze(
        [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]],
        [0.0, 1.0, 2.0, 3.0],
        ["ascending", "descending"],
    )
    if report.backend is None or report.backend.name != "v1-metal-cabi":
        raise AssertionError(f"installed Metal payload resolved to {report.backend!r}")
    if not report.backend.is_gpu or not list(report.interactions):
        raise AssertionError(
            "installed Metal public API did not produce GPU interaction results"
        )
    if (
        report.backend.requested_precision != "fp32"
        or report.backend.effective_precision != "fp32"
        or report.backend.storage_dtype != "float32"
        or report.backend.interaction_arithmetic != "float32"
        or report.backend.reduction_dtype != "float32"
        or report.backend.result_dtype != "float32"
    ):
        raise AssertionError(
            f"installed Metal did not report the full fp32 lane: {report.backend!r}"
        )
    for unsupported in ("mixed", "fp64"):
        try:
            gafime.GafimeEngine(
                gafime.EngineConfig(backend="metal", precision=unsupported)
            ).analyze([[0.0], [1.0]], [0.0, 1.0], ["x"])
        except Exception as exc:
            message = str(exc)
            if "Metal supports precision='fp32' only" not in message:
                raise AssertionError(
                    f"Metal precision={unsupported!r} rejection was not actionable: "
                    f"{message!r}"
                ) from exc
        else:
            raise AssertionError(
                f"installed Metal accepted unsupported precision={unsupported!r}"
            )


def _exercise_distributed_profiles(gafime: object, backend: str) -> None:
    """Execute every advertised profile through the installed top-level API."""

    if backend == "metal":
        _exercise_metal_public_api(gafime)
        return
    features = [[float(index), float((index * 5 + 1) % 17)] for index in range(32)]
    target = [float((index * 7 + 3) % 19) for index in range(32)]
    for profile in EXPECTED_PROFILES[backend]:
        report = gafime.GafimeEngine(
            gafime.EngineConfig(
                backend=backend,
                precision=profile,
                metric_names=("pearson", "spearman", "mutual_info", "r2"),
                permutation_tests=0,
                num_repeats=1,
                budget=gafime.ComputeBudget(
                    max_comb_size=1,
                    max_combinations_per_k=8,
                ),
            )
        ).analyze(features, target, ["trend", "cycle"])
        if report.backend is None or report.backend.selected_backend != backend:
            raise AssertionError(
                f"installed {backend} precision={profile!r} resolved to "
                f"{report.backend!r}"
            )
        expected_storage = "float64" if profile == "fp64" else "float32"
        expected_result = "float32" if profile == "fp32" else "float64"
        expected = (
            profile,
            profile,
            expected_storage,
            expected_storage,
            expected_result,
            expected_result,
        )
        actual = (
            report.backend.requested_precision,
            report.backend.effective_precision,
            report.backend.storage_dtype,
            report.backend.interaction_arithmetic,
            report.backend.reduction_dtype,
            report.backend.result_dtype,
        )
        if actual != expected:
            raise AssertionError(
                f"installed {backend} precision={profile!r} domains "
                f"{actual!r} != {expected!r}"
            )
        rows = list(report.interactions)
        if not rows or any(
            set(row.metrics) != {"pearson", "spearman", "mutual_info", "r2"}
            for row in rows
        ):
            raise AssertionError(
                f"installed {backend} precision={profile!r} did not execute all metrics"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=tuple(PAYLOADS),
        required=True,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="checkout root that must not provide imported packages",
    )
    parser.add_argument(
        "--execute-metal",
        action="store_true",
        help="execute the top-level Metal public API after artifact checks",
    )
    parser.add_argument(
        "--execute-profiles",
        action="store_true",
        help="physically query and execute every profile supported by the backend",
    )
    args = parser.parse_args()
    if args.execute_metal and args.backend != "metal":
        parser.error("--execute-metal is only valid with --backend metal")

    backend = args.backend
    source_root = (args.source_root or Path.cwd()).resolve()
    _remove_checkout_paths(source_root)
    payload = PAYLOADS[args.backend]
    os.environ.pop(payload["env"], None)
    if backend == "metal":
        os.environ.pop("GAFIME_METAL_V1_METALLIB", None)

    gafime = importlib.import_module("gafime")
    _assert_installed(gafime, source_root, "gafime")
    expected_version = str(getattr(gafime, "__version__", ""))
    if importlib.metadata.version("gafime") != expected_version:
        raise AssertionError(
            "installed gafime distribution and package versions differ"
        )
    payload_module = None
    if backend != "metal":
        payload_module = importlib.import_module(payload["package"])
        _assert_installed(payload_module, source_root, payload["package"])
        if importlib.metadata.version(payload["distribution"]) != expected_version:
            raise AssertionError(
                f"{payload['distribution']} does not match installed gafime version "
                f"{expected_version}"
            )
    rocm_policy = None
    if backend == "cuda":
        assert payload_module is not None
        _assert_cuda_build_policy(payload_module)
    elif backend == "rocm":
        assert payload_module is not None
        rocm_policy = _assert_rocm_build_policy(payload_module, source_root)

    discovery = importlib.import_module("gafime._payloads")
    discovered = discovery.discover_payloads(backend)
    library_value = os.environ.get(payload["env"])
    if not library_value:
        raise AssertionError(f"installed {args.backend} payload was not discovered")
    library = Path(library_value).resolve()
    if not library.is_file():
        raise AssertionError(f"discovered payload library does not exist: {library}")
    if backend != "metal":
        assert payload_module is not None
        package_path = Path(payload_module.__file__).resolve().parent
        if not _is_within(library, package_path):
            raise AssertionError(
                f"discovery selected {library}, not the installed "
                f"{payload['package']} package"
            )
        _assert_package_helpers(payload_module, package_path, library)
    if backend == "rocm":
        if rocm_policy is None:
            raise AssertionError("ROCm policy was not loaded")
        _assert_linux_rocm_runtime(library, package_path, rocm_policy)
    elif backend == "cuda":
        _assert_cuda_distribution_surface(library)
    elif backend == "metal":
        metallib_value = os.environ.get("GAFIME_METAL_V1_METALLIB")
        if not metallib_value:
            raise AssertionError(
                "Metal discovery did not configure its paired metallib"
            )
        metallib = Path(metallib_value).resolve()
        if not metallib.is_file() or metallib.parent != library.parent:
            raise AssertionError(
                "Metal discovery did not select a paired dylib/metallib"
            )

    if backend not in discovered:
        raise AssertionError(f"discovery did not report the {backend} payload")
    if backend == "rocm":
        capabilities = gafime.backend_capabilities("rocm", probe=False)
        if capabilities.payload_build_policy.value != rocm_policy:
            raise AssertionError(
                "public capabilities do not expose the installed ROCm wheel policy"
            )
        if capabilities.payload_build_policy.source != "package":
            raise AssertionError(
                "installed ROCm wheel policy must report package evidence"
            )
    elif backend == "metal":
        capabilities = gafime.backend_capabilities(
            "metal", probe=False, precision="fp32"
        )
        expected_policy = {
            "distribution_identity": "gafime",
            "packaging": "embedded-in-macos-arm64-core-wheel",
            "library": "libgafime_metal_v1.dylib",
            "metallib": "gafime_metal_v1.metallib",
        }
        if capabilities.payload_build_policy.value != expected_policy:
            raise AssertionError(
                "public capabilities do not expose the bundled Metal wheel policy"
            )
        if capabilities.payload_build_policy.source != "static":
            raise AssertionError(
                "bundled Metal wheel policy must report a static package contract"
            )
    _assert_payload_separation(backend, payload)
    _assert_exported_symbols(library, backend)
    _assert_public_precision_capabilities(gafime, backend, probe=False)
    if args.execute_profiles or args.execute_metal:
        _assert_precision_capability_abi(library, backend)
        _assert_public_precision_capabilities(gafime, backend, probe=True)
    if args.execute_profiles:
        _exercise_distributed_profiles(gafime, backend)
    if args.execute_metal:
        _exercise_metal_public_api(gafime)
    print(
        f"INSTALLED {args.backend.upper()} PAYLOAD: PASS "
        f"version={expected_version} library={library}"
    )


if __name__ == "__main__":
    main()
