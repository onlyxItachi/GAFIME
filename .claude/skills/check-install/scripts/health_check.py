#!/usr/bin/env python3
"""
GAFIME Installation Health Check

Comprehensive verification that GAFIME is correctly installed and functional.
"""

import json
import platform
import sys
import time


def check(name: str, func) -> dict:
    """Run a check and return result."""
    try:
        result = func()
        return {"name": name, "status": "PASS", **result}
    except Exception as e:
        return {"name": name, "status": "FAIL", "error": str(e)}


def check_python_version():
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        raise RuntimeError(f"Python {version_str} is too old. GAFIME requires 3.10+")
    return {"version": version_str}


def check_gafime_import():
    import gafime
    return {"version": gafime.__version__}


def check_numpy():
    import numpy as np
    return {"version": np.__version__}


def check_polars():
    import polars as pl
    return {"version": pl.__version__}


def check_sklearn():
    try:
        from gafime.sklearn import GafimeSelector
        import sklearn
        return {"version": sklearn.__version__, "gafime_selector": True}
    except ImportError:
        return {"status": "SKIP", "note": "Install with: pip install gafime[sklearn]"}


def check_cuda_backend():
    try:
        from gafime.backends.native_cuda_backend import NativeCudaBackend
        backend = NativeCudaBackend(device_id=0)
        info = backend.info()
        return {
            "device": info.device,
            "memory_total_mb": info.memory_total_mb,
            "memory_free_mb": info.memory_free_mb,
        }
    except ImportError:
        return {"status": "SKIP", "note": "CUDA backend not compiled"}
    except Exception as e:
        return {"status": "SKIP", "note": str(e)}


def check_metal_backend():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {"status": "SKIP", "note": "Not macOS arm64"}
    try:
        from gafime.backends.native_metal_backend import NativeMetalBackend
        backend = NativeMetalBackend()
        return {"available": True}
    except Exception as e:
        return {"status": "SKIP", "note": str(e)}


def check_numpy_backend():
    from gafime.backends.base import Backend
    backend = Backend()
    return {"available": True}


def check_functional():
    """Run a tiny end-to-end analysis."""
    import numpy as np
    from gafime import GafimeEngine, EngineConfig, ComputeBudget

    np.random.seed(42)
    n = 500
    f0 = np.random.randn(n).astype(np.float32)
    f1 = np.random.randn(n).astype(np.float32)
    f2 = np.random.randn(n).astype(np.float32)
    y = (0.8 * f0 * f1 + 0.1 * np.random.randn(n)).astype(np.float32)
    X = np.column_stack([f0, f1, f2])

    config = EngineConfig(
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=20),
        permutation_tests=3,
        num_repeats=2,
    )
    engine = GafimeEngine(config)
    report = engine.analyze(X, y, feature_names=["alpha", "beta", "gamma"])

    backend_name = report.backend.name if report.backend else "numpy"

    return {
        "signal_detected": report.decision.signal_detected,
        "interactions_found": len(report.interactions),
        "backend_used": backend_name,
    }


def check_throughput():
    """Quick throughput benchmark."""
    import numpy as np
    from gafime import GafimeEngine, EngineConfig, ComputeBudget

    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 10).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.1 * np.random.randn(n)).astype(np.float32)

    config = EngineConfig(
        budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=200),
        permutation_tests=0,
        num_repeats=1,
    )
    engine = GafimeEngine(config)

    start = time.perf_counter()
    report = engine.analyze(X, y)
    elapsed = time.perf_counter() - start

    n_combos = len(report.interactions)
    throughput = n_combos / elapsed if elapsed > 0 else 0

    return {
        "combinations": n_combos,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_per_sec": round(throughput, 0),
    }


def main():
    checks = [
        check("Python Version", check_python_version),
        check("GAFIME Import", check_gafime_import),
        check("NumPy", check_numpy),
        check("Polars", check_polars),
        check("Scikit-Learn", check_sklearn),
        check("CUDA Backend", check_cuda_backend),
        check("Metal Backend", check_metal_backend),
        check("NumPy Backend", check_numpy_backend),
        check("Functional Test", check_functional),
        check("Throughput Benchmark", check_throughput),
    ]

    # Summary
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    skipped = sum(1 for c in checks if c["status"] == "SKIP")

    report = {
        "checks": checks,
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(checks),
            "all_critical_passed": failed == 0,
        },
    }

    print(json.dumps(report, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
