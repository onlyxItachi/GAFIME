#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import platform
import sys
import tempfile
from pathlib import Path


RELEASE_STATUS = "see_docs_releases_status"


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _validate_distribution_versions(
    installed: dict[str, str | None], *, runtime_version: str | None = None
) -> str:
    """Require every installed vendor payload to match installed Core exactly."""

    core_version = installed.get("gafime")
    if core_version is None:
        raise RuntimeError("the gafime Core distribution is not installed")
    if runtime_version is not None and runtime_version != core_version:
        raise RuntimeError(
            "imported gafime/runtime metadata mismatch: "
            f"runtime={runtime_version}, distribution={core_version}"
        )
    mismatches = {
        name: version
        for name, version in installed.items()
        if name != "gafime" and version is not None and version != core_version
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={version} (Core={core_version})"
            for name, version in sorted(mismatches.items())
        )
        raise RuntimeError(f"payload/Core exact-version mismatch: {detail}")
    return json.dumps(installed, sort_keys=True)


def _validate_python_version(
    version_info: tuple[int, ...], *, implementation: str = "CPython"
) -> str:
    """Require a CPython minor covered by the current v1 release matrix."""

    if implementation != "CPython":
        raise RuntimeError(f"GAFIME v1 supports CPython only; found {implementation}")
    major_minor = tuple(version_info[:2])
    if major_minor not in {(3, minor) for minor in range(10, 15)}:
        raise RuntimeError(
            "GAFIME v1 supports CPython 3.10 through 3.14; "
            f"found {major_minor[0]}.{major_minor[1]}"
        )
    return f"{major_minor[0]}.{major_minor[1]} (release-tested 3.10-3.14)"


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
        _validate_python_version(
            sys.version_info, implementation=platform.python_implementation()
        )
        return f"{platform.python_version()} (release-tested 3.10-3.14)"

    check("Python", python_version)
    check(
        "Release status",
        lambda: (
            f"release_status={RELEASE_STATUS}; consult docs/releases/STATUS.md, "
            "GitHub Releases, and PyPI for mutable publication state"
        ),
    )

    def import_gafime() -> str:
        import gafime

        return f"gafime {gafime.__version__}"

    check("GAFIME", import_gafime)

    def distributions() -> str:
        import gafime

        installed = {
            name: _dist_version(name)
            for name in ("gafime", "gafime-cuda", "gafime-rocm")
        }
        return _validate_distribution_versions(
            installed, runtime_version=gafime.__version__
        )

    check("Exact distribution versions", distributions)

    def capability_snapshot() -> str:
        from gafime import backend_capabilities

        caps = backend_capabilities("auto", probe=True, precision="mixed")
        precision = caps.precision_contract.value
        return (
            f"status={caps.selection_status}, selected={caps.selected_backend}, "
            f"boundary={caps.native_boundary.value}, version={caps.native_version.value}, "
            f"precision={json.dumps(precision, sort_keys=True)}"
        )

    check("Backend capability probe", capability_snapshot)

    def family_contract() -> str:
        from gafime import available_families

        families = available_families()
        names = tuple(family.name for family in families)
        if names != ("continuous", "decision_path", "time_series"):
            raise AssertionError(f"unexpected family registry: {names}")
        decision_path = next(
            family for family in families if family.name == "decision_path"
        )
        if not decision_path.significance_support.permutation:
            raise AssertionError(
                "decision_path must disclose permutation significance support"
            )
        return ", ".join(names) + "; decision_path_permutation=true"

    check("Family capability contract", family_contract)

    def core_engine() -> str:
        from gafime import ComputeBudget, EngineConfig, GafimeEngine

        features = [
            [float(index), float(index % 3), float((index * 5) % 7)]
            for index in range(80)
        ]
        target = [row[0] * row[1] for row in features]
        summaries = []
        for precision in ("fp32", "mixed", "fp64"):
            report = GafimeEngine(
                EngineConfig(
                    backend="core",
                    metric_names=("pearson", "r2"),
                    budget=ComputeBudget(max_comb_size=2, max_combinations_per_k=8),
                    permutation_tests=0,
                    num_repeats=1,
                    precision=precision,
                )
            ).analyze(features, target)
            if report.backend is None or report.backend.selected_backend != "core":
                raise AssertionError(
                    f"unexpected Core report backend: {report.backend!r}"
                )
            if report.backend.effective_precision != precision:
                raise AssertionError(
                    "Core report precision mismatch: "
                    f"requested={precision}, effective={report.backend.effective_precision}"
                )
            summaries.append(f"{precision}={len(report.interactions)}")
        return "interactions: " + ", ".join(summaries)

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
