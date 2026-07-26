from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _cli(
    *arguments: str, module: str = "gafime.cli"
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if environment.get("GAFIME_TEST_INSTALLED_PACKAGE") == "1":
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(ROOT / "python")
    environment.pop("GAFIME_V1_BOUNDARY_MODULE", None)
    for name in ("GAFIME_CUDA_V1_LIB", "GAFIME_ROCM_V1_LIB", "GAFIME_METAL_V1_LIB"):
        environment.pop(name, None)
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_module_preserves_the_legacy_cli_entrypoint():
    result = _cli("--version", module="gafime")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gafime 1.0.0b0"


def test_check_reports_core_package_native_version_and_static_capabilities():
    result = _cli("--check", "--backend", "core")

    assert result.returncode == 0, result.stderr
    assert "GAFIME package: 1.0.0b0" in result.stdout
    assert "native version: 1.0.0b0" in result.stdout
    assert "configured backend: core" in result.stdout
    assert "selected backend: core" in result.stdout
    assert "backend status: available" in result.stdout
    assert "graph support: False" in result.stdout
    assert "family time_series: generation=gafime_cpu" in result.stdout
    assert "family decision_path: generation=gafime_cpu" in result.stdout
    assert "significance=permutation:False,stability:True" in result.stdout


def test_check_reports_explicit_unavailable_backend_without_cpu_substitution():
    result = _cli("--check", "--backend", "cuda")

    assert result.returncode == 1
    assert "configured backend: cuda" in result.stdout
    assert "selected backend: unknown" in result.stdout
    assert "backend status: unavailable" in result.stdout
    assert "candidate cuda: unavailable" in result.stdout
    assert "selected backend: core" not in result.stdout
