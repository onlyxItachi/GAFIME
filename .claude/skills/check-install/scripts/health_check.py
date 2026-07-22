#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import platform
import sys
import tempfile
from pathlib import Path


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    checks: list[tuple[str, str, str]] = []

    def check(name: str, fn, *, required: bool = True) -> None:
        try:
            detail = str(fn())
            checks.append((name, "PASS", detail))
        except Exception as exc:
            status = "FAIL" if required else "SKIP"
            checks.append((name, status, f"{type(exc).__name__}: {exc}"))

    def python_version() -> str:
        if sys.version_info < (3, 10):
            raise RuntimeError("GAFIME requires Python 3.10 or newer")
        return f"{platform.python_version()} (>= 3.10 required)"

    check("Python", python_version)

    def import_gafime() -> str:
        import gafime

        return f"gafime {gafime.__version__}"

    check("GAFIME", import_gafime)

    def distributions() -> str:
        installed = {
            name: _dist_version(name)
            for name in ("gafime", "gafime-cuda", "gafime-rocm", "gafime-cuda-rt")
        }
        return json.dumps(installed, sort_keys=True)

    check("Distribution versions", distributions)

    def capability_snapshot() -> str:
        from gafime import backend_capabilities

        caps = backend_capabilities("auto", probe=True)
        return (
            f"status={caps.selection_status}, selected={caps.selected_backend}, "
            f"boundary={caps.native_boundary.value}, version={caps.native_version.value}"
        )

    check("Backend capability probe", capability_snapshot)

    def family_contract() -> str:
        from gafime import available_families

        families = available_families()
        names = tuple(family.name for family in families)
        if names != ("continuous", "decision_path", "time_series"):
            raise AssertionError(f"unexpected family registry: {names}")
        decision_path = next(family for family in families if family.name == "decision_path")
        if decision_path.significance_support.permutation:
            raise AssertionError("decision_path must disclose unavailable permutation significance")
        return ", ".join(names)

    check("Family capability contract", family_contract)

    def core_engine() -> str:
        from gafime import ComputeBudget, EngineConfig, GafimeEngine

        features = [
            [float(index), float(index % 3), float((index * 5) % 7)]
            for index in range(80)
        ]
        target = [row[0] * row[1] for row in features]
        report = GafimeEngine(
            EngineConfig(
                backend="core",
                metric_names=("pearson", "r2"),
                budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
                permutation_tests=0,
                num_repeats=1,
            )
        ).analyze(features, target)
        if report.backend is None or report.backend.selected_backend != "core":
            raise AssertionError(f"unexpected Core report backend: {report.backend!r}")
        return f"{len(report.interactions)} interactions"

    check("Rust Core analysis", core_engine)

    def tutorial_generator() -> str:
        from gafime import generate_tutorial

        with tempfile.TemporaryDirectory(prefix="gafime-health-") as temp_dir:
            path = Path(generate_tutorial(str(Path(temp_dir) / "tutorial.ipynb")))
            notebook = json.loads(path.read_text(encoding="utf-8"))
        reference = notebook.get("metadata", {}).get("gafime_reference", {})
        if reference.get("release_scope") != "GAFIME v1 public API":
            raise AssertionError(f"unexpected tutorial metadata: {reference!r}")
        return f"{len(notebook['cells'])} cells"

    check("Starter notebook generator", tutorial_generator)

    def optional_import(name: str) -> str:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "available"))

    check("Polars", lambda: optional_import("polars"))
    check("scikit-learn", lambda: optional_import("sklearn"), required=False)

    for name, status, detail in checks:
        print(f"[{status}] {name}: {detail}")
    return 1 if any(status == "FAIL" for _, status, _ in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
