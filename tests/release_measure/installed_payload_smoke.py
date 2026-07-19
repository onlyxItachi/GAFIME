#!/usr/bin/env python3
"""Clean installed-payload discovery, separation, and ABI-export smoke."""

from __future__ import annotations

import argparse
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

CUDA_RT_PAYLOAD = {
    "distribution": "gafime-cuda-rt",
    "package": "gafime_cuda_rt",
    "env": "GAFIME_CUDA_V1_LIB",
}

CUDA_BUILD_POLICY = {
    "cuda_architectures": ["75", "80", "86", "89", "90", "100", "120"],
    "cuda_tuning_policy": "package-wide-sm89",
    "cuda_tuning_sm": 89,
    "per_architecture_tuning": False,
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


def _assert_cuda_build_policy(payload_module: object, expected_rt_mode: str) -> None:
    module_path = Path(str(payload_module.__file__)).resolve().parent
    policy_path = module_path / "build_policy.json"
    if not policy_path.is_file():
        raise AssertionError(
            f"installed CUDA payload has no build policy: {policy_path}"
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    expected = {**CUDA_BUILD_POLICY, "optix_rt": expected_rt_mode}
    if policy != expected:
        raise AssertionError(f"installed CUDA build policy {policy!r} != {expected!r}")


def _assert_payload_separation(backend: str, payload: dict[str, str]) -> None:
    base = importlib.metadata.distribution("gafime")
    base_files = _distribution_files(base)
    vendor_entries = (
        "gafime_cuda/",
        "gafime_cuda_rt/",
        "gafime_rocm/",
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
    vendor_payloads = (PAYLOADS["cuda"], CUDA_RT_PAYLOAD, PAYLOADS["rocm"])
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


def _assert_exported_symbols(library: Path) -> None:
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
        symbol for symbol in REQUIRED_GPU_ABI_SYMBOLS if symbol not in result.stdout
    ]
    if missing:
        raise AssertionError(
            f"payload library is missing required GPU ABI exports: {missing}"
        )


def _exercise_metal_public_api(gafime: object) -> None:
    config = gafime.EngineConfig(
        backend="metal",
        metric_names=("pearson", "r2"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(PAYLOADS), required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="checkout root that must not provide imported packages",
    )
    parser.add_argument(
        "--cuda-rt",
        choices=("off", "on"),
        default="off",
        help="Expected immutable CUDA OptiX build policy.",
    )
    parser.add_argument(
        "--execute-metal",
        action="store_true",
        help="execute the top-level Metal public API after artifact checks",
    )
    args = parser.parse_args()
    if args.execute_metal and args.backend != "metal":
        parser.error("--execute-metal is only valid with --backend metal")
    if args.backend != "cuda" and args.cuda_rt != "off":
        parser.error("--cuda-rt applies only to --backend cuda")

    source_root = (args.source_root or Path.cwd()).resolve()
    _remove_checkout_paths(source_root)
    payload = (
        CUDA_RT_PAYLOAD
        if args.backend == "cuda" and args.cuda_rt == "on"
        else PAYLOADS[args.backend]
    )
    os.environ.pop(payload["env"], None)
    if args.backend == "metal":
        os.environ.pop("GAFIME_METAL_V1_METALLIB", None)

    gafime = importlib.import_module("gafime")
    _assert_installed(gafime, source_root, "gafime")
    expected_version = str(getattr(gafime, "__version__", ""))
    if importlib.metadata.version("gafime") != expected_version:
        raise AssertionError(
            "installed gafime distribution and package versions differ"
        )
    if args.backend != "metal":
        payload_module = importlib.import_module(payload["package"])
        _assert_installed(payload_module, source_root, payload["package"])
        if importlib.metadata.version(payload["distribution"]) != expected_version:
            raise AssertionError(
                f"{payload['distribution']} does not match installed gafime version {expected_version}"
            )
        if args.backend == "cuda":
            _assert_cuda_build_policy(payload_module, args.cuda_rt)

    discovery = importlib.import_module("gafime._payloads")
    discovered = discovery.discover_payloads(args.backend)
    library_value = os.environ.get(payload["env"])
    if not library_value:
        raise AssertionError(f"installed {args.backend} payload was not discovered")
    library = Path(library_value).resolve()
    if not library.is_file():
        raise AssertionError(f"discovered payload library does not exist: {library}")
    if args.backend != "metal":
        package_path = Path(payload_module.__file__).resolve().parent
        if not _is_within(library, package_path):
            raise AssertionError(
                f"discovery selected {library}, not the installed {payload['package']} package"
            )
    else:
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

    if args.backend not in discovered:
        raise AssertionError(f"discovery did not report the {args.backend} payload")
    _assert_payload_separation(args.backend, payload)
    _assert_exported_symbols(library)
    if args.execute_metal:
        _exercise_metal_public_api(gafime)
    print(
        f"INSTALLED {args.backend.upper()} PAYLOAD: PASS "
        f"version={expected_version} library={library}"
    )


if __name__ == "__main__":
    main()
