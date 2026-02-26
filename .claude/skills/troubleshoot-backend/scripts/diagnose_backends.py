#!/usr/bin/env python3
"""
GAFIME Backend Diagnostics

Tests each GAFIME backend individually and reports detailed error information.
"""

import ctypes
import json
import os
import platform
import sys
from pathlib import Path


def find_gafime_package_dir():
    """Locate the installed gafime package directory."""
    try:
        import gafime
        return Path(gafime.__file__).parent
    except ImportError:
        return None


def check_library_file(pkg_dir, names):
    """Check if a shared library file exists in the package directory."""
    if pkg_dir is None:
        return None, "gafime package not found"

    for name in names:
        path = pkg_dir / name
        if path.exists():
            return str(path), None

    # Also check parent and current working directory
    for search_dir in [pkg_dir.parent, Path.cwd()]:
        for name in names:
            path = search_dir / name
            if path.exists():
                return str(path), None

    return None, f"None of {names} found in {pkg_dir}"


def try_load_library(path):
    """Try to load a shared library and return success/error."""
    if path is None:
        return False, "Library file not found"
    try:
        ctypes.CDLL(path)
        return True, None
    except OSError as e:
        return False, str(e)


def diagnose_cuda_backend(pkg_dir):
    """Diagnose CUDA backend."""
    result = {
        "name": "CUDA (native)",
        "status": "UNKNOWN",
        "library_file": None,
        "library_found": False,
        "library_loads": False,
        "python_import": False,
        "functional": False,
        "error": None,
        "details": {},
    }

    # Check library file
    if platform.system() == "Windows":
        lib_names = ["gafime_cuda.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["libgafime_cuda.dylib"]
    else:
        lib_names = ["libgafime_cuda.so"]

    lib_path, find_err = check_library_file(pkg_dir, lib_names)
    result["library_file"] = lib_path
    result["library_found"] = lib_path is not None
    if find_err:
        result["details"]["find_error"] = find_err

    # Try loading the library
    if lib_path:
        loads, load_err = try_load_library(lib_path)
        result["library_loads"] = loads
        if load_err:
            result["details"]["load_error"] = load_err

    # Try Python import
    try:
        from gafime.backends.native_cuda_backend import NativeCudaBackend
        result["python_import"] = True

        try:
            backend = NativeCudaBackend(device_id=0)
            info = backend.info()
            result["functional"] = True
            result["status"] = "OK"
            result["details"]["device"] = info.device
            result["details"]["memory_total_mb"] = info.memory_total_mb
            result["details"]["memory_free_mb"] = info.memory_free_mb
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "LOAD_FAIL"
    except ImportError as e:
        result["error"] = str(e)
        result["status"] = "IMPORT_FAIL"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "ERROR"

    if not result["functional"]:
        if not result["library_found"]:
            result["status"] = "MISSING"
        elif not result["library_loads"]:
            result["status"] = "LOAD_FAIL"

    return result


def diagnose_metal_backend(pkg_dir):
    """Diagnose Metal backend."""
    result = {
        "name": "Metal (Apple Silicon)",
        "status": "UNKNOWN",
        "library_file": None,
        "library_found": False,
        "metallib_found": False,
        "python_import": False,
        "functional": False,
        "error": None,
        "details": {},
    }

    if platform.system() != "Darwin":
        result["status"] = "N/A"
        result["error"] = "Metal is only available on macOS"
        return result

    if platform.machine() != "arm64":
        result["status"] = "N/A"
        result["error"] = "Metal requires Apple Silicon (arm64)"
        return result

    # Check library files
    lib_path, _ = check_library_file(pkg_dir, ["gafime_metal.dylib", "libgafime_metal.dylib"])
    result["library_file"] = lib_path
    result["library_found"] = lib_path is not None

    metallib_path, _ = check_library_file(pkg_dir, ["gafime_kernels.metallib"])
    result["metallib_found"] = metallib_path is not None

    # Try Python import
    try:
        from gafime.backends.native_metal_backend import NativeMetalBackend
        result["python_import"] = True

        try:
            backend = NativeMetalBackend()
            result["functional"] = True
            result["status"] = "OK"
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "LOAD_FAIL"
    except ImportError as e:
        result["error"] = str(e)
        result["status"] = "IMPORT_FAIL"

    if not result["functional"] and not result["library_found"]:
        result["status"] = "MISSING"

    return result


def diagnose_core_backend(pkg_dir):
    """Diagnose C++ core (pybind11) backend."""
    result = {
        "name": "C++ Core (pybind11)",
        "status": "UNKNOWN",
        "module_found": False,
        "python_import": False,
        "functional": False,
        "error": None,
        "details": {},
    }

    # Check for compiled module
    if pkg_dir:
        for ext in ["*.pyd", "*.so", "*.dylib"]:
            for f in pkg_dir.glob(f"gafime_core*{ext.replace('*', '')}"):
                result["module_found"] = True
                result["details"]["module_path"] = str(f)
                break

    # Try Python import
    try:
        from gafime.backends.core_backend import CoreBackend
        result["python_import"] = True
        try:
            backend = CoreBackend()
            result["functional"] = True
            result["status"] = "OK"
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "LOAD_FAIL"
    except (ImportError, ModuleNotFoundError) as e:
        result["error"] = str(e)
        result["status"] = "IMPORT_FAIL"

    if not result["functional"] and not result["module_found"]:
        result["status"] = "MISSING"

    return result


def diagnose_numpy_backend():
    """Diagnose NumPy fallback backend."""
    result = {
        "name": "NumPy (fallback)",
        "status": "OK",
        "python_import": False,
        "functional": False,
        "error": None,
        "details": {},
    }

    try:
        from gafime.backends.base import Backend
        result["python_import"] = True
        backend = Backend()
        result["functional"] = True
        result["status"] = "OK"

        import numpy as np
        result["details"]["numpy_version"] = np.__version__
    except ImportError as e:
        result["error"] = str(e)
        result["status"] = "IMPORT_FAIL"

    return result


def diagnose_resolution():
    """Test the actual backend resolution logic."""
    result = {
        "resolved_backend": None,
        "warnings": [],
        "error": None,
    }

    try:
        import numpy as np
        from gafime.backends import resolve_backend
        from gafime.config import EngineConfig

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        config = EngineConfig()

        backend, warnings = resolve_backend(config, X, y)
        result["resolved_backend"] = backend.name
        result["warnings"] = warnings
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    pkg_dir = find_gafime_package_dir()

    report = {
        "gafime_installed": pkg_dir is not None,
        "gafime_package_dir": str(pkg_dir) if pkg_dir else None,
        "system": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "backends": {
            "cuda": diagnose_cuda_backend(pkg_dir),
            "metal": diagnose_metal_backend(pkg_dir),
            "core": diagnose_core_backend(pkg_dir),
            "numpy": diagnose_numpy_backend(),
        },
        "resolution": diagnose_resolution(),
    }

    # Summary
    working = [name for name, info in report["backends"].items() if info["status"] == "OK"]
    failing = [name for name, info in report["backends"].items() if info["status"] not in ("OK", "N/A", "MISSING")]

    report["summary"] = {
        "working_backends": working,
        "failing_backends": failing,
        "best_available": report["resolution"]["resolved_backend"],
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
