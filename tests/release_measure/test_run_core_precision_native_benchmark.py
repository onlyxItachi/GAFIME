"""Adversarial tests for the standalone Core benchmark compiler/runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


_SCRIPT = Path(__file__).with_name("run_core_precision_native_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("gafime_core_native_runner", _SCRIPT)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def test_parser_defaults_to_pinned_release_toolchain() -> None:
    args = runner._parser().parse_args(
        [
            "--product-source-root",
            "/product",
            "--harness-source-root",
            "/harness",
            "--product-rlib",
            "/product/libgafime_cpu.rlib",
            "--wheel",
            "/product/gafime.whl",
            "--binary",
            "/evidence/benchmark",
            "--output",
            "/evidence/report.json",
            "--input-policy",
            "common-f64",
        ]
    )

    assert args.toolchain == "1.97.1"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path, source: bytes) -> Path:
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "benchmark-test@example.invalid")
    _git(root, "config", "user.name", "Benchmark Test")
    path = root / runner.HARNESS_SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    runner_path = root / runner.HARNESS_RUNNER
    runner_path.write_bytes(b"# tracked runner fixture\n")
    _git(
        root,
        "add",
        runner.HARNESS_SOURCE.as_posix(),
        runner.HARNESS_RUNNER.as_posix(),
    )
    _git(root, "commit", "-m", "test fixture")
    return path


def test_tracked_harness_source_rejects_uncommitted_drift(tmp_path: Path) -> None:
    root = tmp_path / "harness"
    source = _repository(root, b"fn main() {}\n")
    clean = runner._repository_identity(root)
    identity = runner._tracked_source_identity(clean.root, runner.HARNESS_SOURCE)
    assert identity.path == source.resolve()

    source.write_bytes(b'fn main() { panic!("drift") }\n')
    with pytest.raises(ValueError, match="checked-in HEAD blob"):
        runner._tracked_source_identity(clean.root, runner.HARNESS_SOURCE)
    with pytest.raises(ValueError, match="repository must be clean"):
        runner._repository_identity(root)


def test_compile_command_uses_harness_source_not_product_source(tmp_path: Path) -> None:
    product_root = tmp_path / "product"
    harness_root = tmp_path / "harness"
    product_source = _repository(product_root, b'compile_error!("product drift");\n')
    harness_source = _repository(harness_root, b"fn main() {}\n")
    product = runner._repository_identity(product_root)
    harness = runner._repository_identity(harness_root)
    source = runner._tracked_source_identity(harness.root, runner.HARNESS_SOURCE)
    product_rlib = product.root / "target/release/deps/libgafime_cpu-product.rlib"
    binary = tmp_path / "benchmark"

    command = runner._compiler_command(
        rustup="rustup",
        toolchain="1.89.0",
        source=source.path,
        product_rlib=product_rlib,
        binary=binary,
    )

    assert str(harness_source.resolve()) in command
    assert str(product_source.resolve()) not in command
    assert f"gafime_cpu={product_rlib}" in command
    assert "cargo" not in command
    assert product.commit != harness.commit


def test_tracked_harness_runner_rejects_uncommitted_drift(tmp_path: Path) -> None:
    root = tmp_path / "harness"
    _repository(root, b"fn main() {}\n")
    clean = runner._repository_identity(root)
    runner_source = runner._tracked_source_identity(clean.root, runner.HARNESS_RUNNER)
    assert runner_source.relative_path == runner.HARNESS_RUNNER

    runner_source.path.write_bytes(b"# changed compiler flags\n")
    with pytest.raises(ValueError, match="checked-in HEAD blob"):
        runner._tracked_source_identity(clean.root, runner.HARNESS_RUNNER)


def test_compile_command_records_release_codegen_policy(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.rs"
    product_rlib = tmp_path / "libgafime_cpu.rlib"
    binary = tmp_path / "benchmark"
    command = runner._compiler_command(
        rustup="rustup",
        toolchain="1.97.1",
        source=source,
        product_rlib=product_rlib,
        binary=binary,
    )
    for expected in (
        "-Copt-level=3",
        "-Ccodegen-units=1",
        "-Clto=fat",
        "-Cembed-bitcode=yes",
    ):
        assert expected in command


def test_compile_environment_embeds_actual_source_and_rlib_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "harness"
    source_path = _repository(root, b"fn main() {}\n")
    repository = runner._repository_identity(root)
    source = runner._tracked_source_identity(repository.root, runner.HARNESS_SOURCE)
    runner_source = runner._tracked_source_identity(
        repository.root, runner.HARNESS_RUNNER
    )
    product_rlib = tmp_path / "libgafime_cpu-test.rlib"
    product_rlib.write_bytes(b"exact product library")

    command = runner._compiler_command(
        rustup="rustup",
        toolchain="1.89.0",
        source=source.path,
        product_rlib=product_rlib,
        binary=tmp_path / "benchmark",
    )
    environment = runner._compiler_environment(
        source, runner_source, product_rlib, command
    )

    assert environment["GAFIME_COMPILED_HARNESS_SOURCE_SHA256"] == source.sha256
    assert environment["GAFIME_COMPILED_HARNESS_SOURCE_GIT_BLOB"] == source.git_blob
    assert (
        environment["GAFIME_COMPILED_HARNESS_SOURCE_RELATIVE_PATH"]
        == runner.HARNESS_SOURCE.as_posix()
    )
    assert environment["GAFIME_COMPILED_PRODUCT_RLIB_SHA256"] == runner._sha256(
        product_rlib
    )
    assert environment["GAFIME_COMPILED_HARNESS_RUNNER_SHA256"] == runner_source.sha256
    command_json = runner.json.dumps(command, ensure_ascii=True, separators=(",", ":"))
    assert (
        environment["GAFIME_COMPILED_COMMAND_SHA256"]
        == runner.hashlib.sha256(command_json.encode("utf-8")).hexdigest()
    )
    assert len(environment["GAFIME_COMPILED_COMMAND_SHA256"]) == 64
    assert source.path == source_path.resolve()
