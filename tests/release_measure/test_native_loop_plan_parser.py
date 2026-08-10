from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = Path(__file__).with_name("native_loop_plan.py")
SPEC = importlib.util.spec_from_file_location(
    "gafime_native_loop_plan_parser_test", SCRIPT
)
assert SPEC and SPEC.loader
native_loop_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native_loop_plan)


def _calibration(path: Path, variant: str, commit: str) -> Path:
    git_path = (
        Path("/usr/bin/git") if Path("/usr/bin/git").is_file() else Path("/bin/git")
    )
    git_digest = hashlib.sha256(git_path.read_bytes()).hexdigest()
    clean_tree = {"status": "clean", "entry_count": 0, "entries": []}
    source_blob = {
        "path": "/tmp/gafime-product",
        "relative_path": "tests/gpu/helper.cpp",
        "source_sha256": "d" * 64,
        "current_git_blob": "e" * 40,
        "head_git_blob": "e" * 40,
    }
    harness_blob = {
        "path": "/tmp/gafime-harness",
        "relative_path": "tests/gpu/helper.cpp",
        "source_sha256": "f" * 64,
        "current_git_blob": "1" * 40,
        "head_git_blob": "1" * 40,
    }
    payload = {
        "schema": native_loop_plan.CALIBRATION_SCHEMA,
        "status": "calibration_only",
        "backend": "cuda",
        "artifact_kind": "cuda_events",
        "evidence_lane": "supplemental_internal_kernel",
        "device": {"name": "test-gpu"},
        "scope_id": "cuda|w|256|8|8|1|32|2|common-f64|supplemental_internal_kernel|cuda_events|test-gpu",
        "variant": variant,
        "source_commit": commit,
        "product_source_commit": commit,
        "harness_source_commit": "3" * 40,
        "workload": {
            "name": "w",
            "rows": 256,
            "features": 8,
            "candidates": 8,
            "arity": 1,
            "mi_bins": 32,
            "top_k": 2,
        },
        "input_policy": "common-f64",
        "input_identity": {"matrix_sha256": "a" * 64},
        "command_line": ["helper", "--calibration-only"],
        "provenance": {
            "benchmark_binary": {"path": "/tmp/helper", "sha256": "c" * 64},
            "payload": None,
            "wheel": None,
        },
        "source_root": "/tmp/gafime-product",
        "product_source_root": "/tmp/gafime-product",
        "harness_source_root": "/tmp/gafime-harness",
        "source_tree_state": clean_tree,
        "product_source_tree_state": clean_tree,
        "harness_source_tree_state": clean_tree,
        "source_blob": source_blob,
        "harness_source_blob": harness_blob,
        "git": {
            "path": str(git_path.resolve()),
            "sha256": git_digest,
            "version": "git version test",
            "git_dir": "/tmp/gafime-product/.git",
            "git_common_dir": "/tmp/gafime-product/.git",
            "removed_environment": ["GIT_DIR"],
        },
        "entries": [{"key": "direct/fp32/pearson/pearson", "loop_count": 4}],
        "entry_count": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def parser_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native parser adversaries")
    output = tmp_path_factory.mktemp("native-loop-plan-parser") / "parser"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "tests/gpu"),
            str(ROOT / "tests/gpu/native_loop_plan_parser_test.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    return output


def _plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    calibration = [
        _calibration(tmp_path / "baseline.json", "baseline", "1" * 40),
        _calibration(tmp_path / "candidate.json", "candidate", "2" * 40),
    ]
    path = tmp_path / "plan.json"
    payload = native_loop_plan.make_plan(calibration, plan_path=path)
    native_loop_plan.write_plan(path, payload)
    return path, payload


def _run(
    binary: Path,
    plan: Path,
    semantic_digest: str,
    file_digest: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), str(plan), semantic_digest, file_digest],
        text=True,
        capture_output=True,
        cwd=ROOT,
    )


def test_native_parser_accepts_real_generator_output_and_rejects_adversaries(
    parser_binary: Path, tmp_path: Path
) -> None:
    plan, payload = _plan(tmp_path)
    semantic_digest = str(payload["plan_sha256"])
    file_digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert _run(parser_binary, plan, semantic_digest, file_digest).returncode == 0

    unknown = json.loads(plan.read_text(encoding="utf-8"))
    unknown["unexpected"] = True
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(
        json.dumps(unknown, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert (
        _run(parser_binary, unknown_path, semantic_digest, file_digest).returncode != 0
    )

    nested_unknown = json.loads(plan.read_text(encoding="utf-8"))
    nested_unknown["scope"]["unexpected"] = 1  # type: ignore[index]
    nested_path = tmp_path / "nested-unknown.json"
    nested_path.write_text(
        json.dumps(nested_unknown, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert (
        _run(parser_binary, nested_path, semantic_digest, file_digest).returncode != 0
    )

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        plan.read_text(encoding="utf-8").replace(
            '"schema":"gafime.native-loop-plan.v1",',
            '"schema":"gafime.native-loop-plan.v1","schema":"gafime.native-loop-plan.v1",',
            1,
        ),
        encoding="utf-8",
    )
    assert (
        _run(parser_binary, duplicate_path, semantic_digest, file_digest).returncode
        != 0
    )

    trailing_path = tmp_path / "trailing.json"
    trailing_path.write_text(plan.read_text(encoding="utf-8") + "{}", encoding="utf-8")
    assert (
        _run(parser_binary, trailing_path, semantic_digest, file_digest).returncode != 0
    )

    assert _run(parser_binary, plan, "0" * 64, file_digest).returncode != 0
