from __future__ import annotations

import json
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

_SCRIPT = Path(__file__).with_name("native_loop_plan.py")
_SPEC = importlib.util.spec_from_file_location("gafime_native_loop_plan", _SCRIPT)
assert _SPEC and _SPEC.loader
native_loop_plan = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = native_loop_plan
_SPEC.loader.exec_module(native_loop_plan)

CALIBRATION_SCHEMA = native_loop_plan.CALIBRATION_SCHEMA
DuplicateJsonKeyError = native_loop_plan.DuplicateJsonKeyError
make_plan = native_loop_plan.make_plan
strict_load = native_loop_plan.strict_load
validate_plan = native_loop_plan.validate_plan
write_plan = native_loop_plan.write_plan


def _calibration(
    path: Path,
    *,
    variant: str,
    commit: str,
    entries: list[dict[str, object]] | None = None,
    scope_id: str = "cuda|w|256|8|8|1|32|2|common-f64|lane|kind|device",
    lane: str = "supplemental_internal_kernel",
    artifact_kind: str = "cuda_events",
) -> Path:
    git_path = (
        Path("/usr/bin/git") if Path("/usr/bin/git").is_file() else Path("/bin/git")
    )
    clean_tree = {"status": "clean", "entry_count": 0, "entries": []}
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "status": "calibration_only",
        "backend": "cuda",
        "artifact_kind": artifact_kind,
        "evidence_lane": lane,
        "device": {"name": "test-gpu"},
        "scope_id": scope_id,
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
        "input_identity": {"matrix_sha256": "a" * 64, "target_sha256": "b" * 64},
        "command_line": ["native-helper", "--calibration-only"],
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
        "source_blob": {
            "path": "/tmp/gafime-product",
            "relative_path": "tests/gpu/helper.cpp",
            "source_sha256": "d" * 64,
            "current_git_blob": "e" * 40,
            "head_git_blob": "e" * 40,
        },
        "harness_source_blob": {
            "path": "/tmp/gafime-harness",
            "relative_path": "tests/gpu/helper.cpp",
            "source_sha256": "f" * 64,
            "current_git_blob": "1" * 40,
            "head_git_blob": "1" * 40,
        },
        "git": {
            "path": str(git_path.resolve()),
            "sha256": hashlib.sha256(git_path.read_bytes()).hexdigest(),
            "version": "git version test",
            "git_dir": "/tmp/gafime-product/.git",
            "git_common_dir": "/tmp/gafime-product/.git",
            "removed_environment": ["GIT_DIR"],
        },
        "entries": entries
        or [
            {"key": "direct/fp32/pearson/pearson", "loop_count": 4},
            {"key": "payload/fp64/execute/pearson", "loop_count": 8},
        ],
    }
    payload["entry_count"] = len(payload["entries"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pair(tmp_path: Path, **kwargs: object) -> list[Path]:
    return [
        _calibration(
            tmp_path / "baseline.json", variant="baseline", commit="1" * 40, **kwargs
        ),
        _calibration(
            tmp_path / "candidate.json", variant="candidate", commit="2" * 40, **kwargs
        ),
    ]


def test_plan_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    plan = make_plan(_pair(tmp_path), plan_path=tmp_path / "first.json")
    assert validate_plan(plan) == []
    assert [item["loop_count"] for item in plan["entries"]] == [8, 16]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_plan(first, plan)
    write_plan(
        second,
        make_plan(list(reversed(_pair(tmp_path))), plan_path=tmp_path / "second.json"),
    )
    assert first.read_bytes() == second.read_bytes()
    assert validate_plan(json.loads(first.read_text(encoding="utf-8"))) == []


def test_calibration_ceiling_retains_the_fixed_plan_headroom(
    tmp_path: Path,
) -> None:
    calibration_ceiling = native_loop_plan.CALIBRATION_LOOP_COUNT_CEILING
    assert native_loop_plan.DEFAULT_HEADROOM_FACTOR == 2
    assert native_loop_plan.DEFAULT_MAX_LOOP_COUNT == calibration_ceiling * 2
    paths = _pair(
        tmp_path,
        entries=[
            {
                "key": "host/fp32/candidate_materialization/",
                "loop_count": calibration_ceiling,
            }
        ],
    )
    plan = make_plan(paths)
    assert plan["max_loop_count"] == calibration_ceiling * 2
    assert plan["entries"] == [
        {
            "key": "host/fp32/candidate_materialization/",
            "loop_count": calibration_ceiling * 2,
        }
    ]


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "x", "schema": "y"}', encoding="utf-8")
    with pytest.raises(DuplicateJsonKeyError):
        strict_load(path)


def test_unequal_key_sets_are_rejected(tmp_path: Path) -> None:
    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["entries"].pop()
    payload["entry_count"] = len(payload["entries"])
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="key sets"):
        make_plan(paths)


def test_scope_and_variant_misuse_are_rejected(tmp_path: Path) -> None:
    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["scope_id"] = "cuda|different-scope"
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scopes"):
        make_plan(paths)
    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["variant"] = "baseline"
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly baseline"):
        make_plan(paths)


def test_same_commit_and_cap_overflow_are_rejected(tmp_path: Path) -> None:
    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["source_commit"] = "1" * 40
    payload["product_source_commit"] = "1" * 40
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="commits"):
        make_plan(paths)
    paths = _pair(
        tmp_path,
        entries=[
            {
                "key": "direct/fp32/pearson/pearson",
                "loop_count": native_loop_plan.CALIBRATION_LOOP_COUNT_CEILING + 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="calibration ceiling"):
        make_plan(paths)


def test_fixed_headroom_and_plan_ceiling_cannot_be_overridden(tmp_path: Path) -> None:
    paths = _pair(tmp_path)
    with pytest.raises(ValueError, match="headroom_factor must equal"):
        make_plan(paths, headroom_factor=1)
    with pytest.raises(ValueError, match="max_loop_count must equal"):
        make_plan(paths, max_loop_count=1 << 20)


def test_product_and_common_harness_commit_bindings_are_required(
    tmp_path: Path,
) -> None:
    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["product_source_commit"] = "4" * 40
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_commit"):
        make_plan(paths)

    paths = _pair(tmp_path)
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    payload["harness_source_commit"] = "4" * 40
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="harness commits"):
        make_plan(paths)


def test_digest_tamper_and_noncanonical_plan_fail_validation(tmp_path: Path) -> None:
    plan = make_plan(_pair(tmp_path))
    tampered = dict(plan)
    tampered["entries"] = [
        dict(plan["entries"][0], loop_count=99),
        *plan["entries"][1:],
    ]
    assert "plan_sha256_mismatch" in validate_plan(tampered)
    malformed = dict(plan)
    malformed["entries"] = [*plan["entries"], dict(plan["entries"][0])]
    assert "entry_2_duplicate_key" in validate_plan(malformed)
    wrong_factor = dict(plan, headroom_factor=1)
    assert "headroom_factor_invalid" in validate_plan(wrong_factor)
    wrong_cap = dict(plan, max_loop_count=1 << 20)
    assert "max_loop_count_invalid" in validate_plan(wrong_cap)
