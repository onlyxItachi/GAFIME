#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn):
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    check("Python", lambda: platform.python_version())

    def import_gafime() -> str:
        import gafime

        return f"gafime {gafime.__version__}"

    check("GAFIME", import_gafime)

    def subfunctions() -> str:
        from gafime import subfunctions

        assert hasattr(subfunctions, "BatchScheduler")
        return f"subfunctions {getattr(subfunctions, '__version__', 'unknown')}"

    check("Rust subfunctions", subfunctions)

    def core_backend() -> str:
        from gafime import ComputeBudget, EngineConfig, GafimeEngine

        X = [[float(i), float(i % 3), float((i * 5) % 7)] for i in range(80)]
        y = [row[0] * row[1] for row in X]
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
                permutation_tests=1,
                num_repeats=1,
            )
        ).analyze(X, y)
        return f"{report.backend.name}, {len(report.interactions)} interactions"

    check("C++ Core engine", core_backend)

    def polars() -> str:
        mod = importlib.import_module("polars")
        return getattr(mod, "__version__", "available")

    check("Polars", polars)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
