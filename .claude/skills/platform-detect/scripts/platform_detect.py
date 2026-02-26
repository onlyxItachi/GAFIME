#!/usr/bin/env python3
"""
GAFIME Platform Detection Script

Detects hardware capabilities and installed GAFIME backends.
Outputs a JSON report with recommended EngineConfig values.
"""

import json
import os
import platform
import sys
from pathlib import Path


def detect_os():
    """Detect OS and architecture."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def detect_cuda():
    """Detect NVIDIA CUDA GPU availability."""
    result = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "cuda_version": None,
        "driver_version": None,
    }

    # Method 1: Try via ctypes (doesn't require PyTorch/CuPy)
    try:
        import ctypes

        if platform.system() == "Windows":
            try:
                cuda_rt = ctypes.CDLL("cudart64_12.dll")
            except OSError:
                try:
                    cuda_rt = ctypes.CDLL("cudart64_110.dll")
                except OSError:
                    cuda_rt = None
        else:
            try:
                cuda_rt = ctypes.CDLL("libcudart.so")
            except OSError:
                cuda_rt = None

        if cuda_rt:
            device_count = ctypes.c_int(0)
            cuda_rt.cudaGetDeviceCount(ctypes.byref(device_count))
            result["device_count"] = device_count.value
            if device_count.value > 0:
                result["available"] = True
    except Exception:
        pass

    # Method 2: Try nvidia-smi for detailed info
    try:
        import subprocess

        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0:
            result["available"] = True
            for i, line in enumerate(smi.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    vram_mb = int(float(parts[1]))
                    result["devices"].append({
                        "id": i,
                        "name": parts[0],
                        "vram_mb": vram_mb,
                        "compute_capability": parts[3],
                    })
                    result["driver_version"] = parts[2] if len(parts) > 2 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: Try nvcc for toolkit version
    try:
        import subprocess

        nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.4"
                    parts = line.split("release")[-1].strip().split(",")[0].strip()
                    result["cuda_version"] = parts
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


def detect_metal():
    """Detect Apple Metal GPU availability."""
    result = {"available": False, "device_name": None}

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return result

    try:
        import subprocess

        sp = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        if sp.returncode == 0 and "Metal" in sp.stdout:
            result["available"] = True
            for line in sp.stdout.split("\n"):
                if "Chipset Model" in line or "Chip" in line:
                    result["device_name"] = line.split(":")[-1].strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


def detect_cpu():
    """Detect CPU capabilities."""
    result = {
        "cores_physical": os.cpu_count(),
        "cores_logical": os.cpu_count(),
        "openmp_available": False,
    }

    try:
        import multiprocessing
        result["cores_logical"] = multiprocessing.cpu_count()
    except Exception:
        pass

    # Check OpenMP by trying to find the runtime
    try:
        import ctypes
        if platform.system() == "Windows":
            ctypes.CDLL("vcomp140.dll")
            result["openmp_available"] = True
        elif platform.system() == "Darwin":
            try:
                ctypes.CDLL("libomp.dylib")
                result["openmp_available"] = True
            except OSError:
                pass
        else:
            ctypes.CDLL("libgomp.so.1")
            result["openmp_available"] = True
    except OSError:
        pass

    return result


def detect_gafime_backends():
    """Check which GAFIME backends are installed and loadable."""
    backends = {
        "gafime_installed": False,
        "gafime_version": None,
        "cuda_backend": False,
        "metal_backend": False,
        "core_backend": False,
        "numpy_backend": True,  # Always available
        "sklearn_available": False,
    }

    try:
        import gafime
        backends["gafime_installed"] = True
        backends["gafime_version"] = getattr(gafime, "__version__", "unknown")
    except ImportError:
        return backends

    # Test CUDA backend
    try:
        from gafime.backends.native_cuda_backend import NativeCudaBackend
        NativeCudaBackend(device_id=0)
        backends["cuda_backend"] = True
    except Exception:
        pass

    # Test Metal backend
    try:
        from gafime.backends.native_metal_backend import NativeMetalBackend
        NativeMetalBackend()
        backends["metal_backend"] = True
    except Exception:
        pass

    # Test C++ core backend
    try:
        from gafime.backends.core_backend import CoreBackend
        CoreBackend()
        backends["core_backend"] = True
    except Exception:
        pass

    # Test sklearn
    try:
        from gafime.sklearn import GafimeSelector
        backends["sklearn_available"] = True
    except ImportError:
        pass

    return backends


def recommend_config(cuda_info, metal_info, cpu_info, backends):
    """Generate recommended EngineConfig based on detected hardware."""
    config = {
        "backend": "auto",
        "device_id": 0,
        "vram_budget_mb": 4096,
        "keep_in_vram": True,
        "recommended_max_comb_size": 2,
        "recommended_max_combinations_per_k": 5000,
    }

    if backends.get("cuda_backend") and cuda_info.get("available"):
        config["backend"] = "cuda"
        if cuda_info.get("devices"):
            gpu = cuda_info["devices"][0]
            config["device_id"] = gpu["id"]
            # Use 75% of VRAM to leave headroom
            config["vram_budget_mb"] = int(gpu["vram_mb"] * 0.75)
            # Larger VRAM → can afford higher-order combinations
            if gpu["vram_mb"] >= 16384:
                config["recommended_max_comb_size"] = 4
                config["recommended_max_combinations_per_k"] = 10000
            elif gpu["vram_mb"] >= 8192:
                config["recommended_max_comb_size"] = 3
                config["recommended_max_combinations_per_k"] = 7500
    elif backends.get("metal_backend") and metal_info.get("available"):
        config["backend"] = "metal"
        config["keep_in_vram"] = True
        # Apple Silicon unified memory — use conservative budget
        config["vram_budget_mb"] = 4096
    else:
        config["backend"] = "cpu" if backends.get("core_backend") else "numpy"
        config["keep_in_vram"] = False
        config["vram_budget_mb"] = 0
        # CPU is slower, keep budgets conservative
        config["recommended_max_combinations_per_k"] = 2000

    return config


def main():
    report = {
        "os": detect_os(),
        "cuda": detect_cuda(),
        "metal": detect_metal(),
        "cpu": detect_cpu(),
        "gafime_backends": detect_gafime_backends(),
    }
    report["recommended_config"] = recommend_config(
        report["cuda"], report["metal"], report["cpu"], report["gafime_backends"],
    )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
