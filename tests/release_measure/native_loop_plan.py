#!/usr/bin/env python3
"""Build and validate the immutable native timing loop plan.

The native CUDA and ROCm helpers calibrate in fresh processes.  Their
calibration-only output is deliberately data, not performance evidence.  This
module combines those outputs into one deterministic plan consumed by every
recorded helper process.  A recorded helper must use exactly the plan count
for every semantic cell; it may not adapt its loop count after the plan has
been installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "gafime.native-loop-plan.v1"
CALIBRATION_SCHEMA = "gafime.native-loop-calibration.v1"
DEFAULT_HEADROOM_FACTOR = 2
CALIBRATION_LOOP_COUNT_CEILING = 1 << 20
# Calibration helpers may legitimately stop at their bounded search ceiling.
# The immutable recorded plan must still have room for the mandatory fixed 2x
# guard band; recorded helpers independently reject a region below 5 ms.
DEFAULT_MAX_LOOP_COUNT = CALIBRATION_LOOP_COUNT_CEILING * DEFAULT_HEADROOM_FACTOR
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
TRUSTED_GIT_PATHS = (Path("/usr/bin/git"), Path("/bin/git"))


def _require_git_identity(payload: Mapping[str, Any], path: Path) -> list[str]:
    failures: list[str] = []
    git = payload.get("git", payload.get("git_identity"))
    if not isinstance(git, Mapping):
        return [f"{path}: Git executable identity is required"]
    git_path = git.get("path")
    trusted_paths = {
        candidate.resolve() for candidate in TRUSTED_GIT_PATHS if candidate.exists()
    }
    if (
        not isinstance(git_path, str)
        or not Path(git_path).is_absolute()
        or Path(git_path).resolve() not in trusted_paths
    ):
        failures.append(f"{path}: absolute Git executable path is required")
    if (
        not isinstance(git.get("sha256"), str)
        or HEX64.fullmatch(str(git.get("sha256"))) is None
        or (
            isinstance(git_path, str)
            and Path(git_path).is_file()
            and _sha256(Path(git_path)) != str(git.get("sha256"))
        )
    ):
        failures.append(f"{path}: Git executable sha256 is required")
    if not isinstance(git.get("version"), str) or not git.get("version"):
        failures.append(f"{path}: Git executable version is required")
    removed = git.get("removed_environment")
    if not isinstance(removed, list) or any(
        not isinstance(name, str) or not name.startswith("GIT_") for name in removed
    ):
        failures.append(f"{path}: removed Git environment names are required")
    for name in ("git_dir", "git_common_dir"):
        if (
            not isinstance(git.get(name), str)
            or not Path(str(git.get(name))).is_absolute()
        ):
            failures.append(f"{path}: absolute {name} is required")
    return failures


def _require_source_binding(
    payload: Mapping[str, Any], name: str, path: Path
) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        return [f"{path}: {name} is required"]
    failures: list[str] = []
    relative = value.get("relative_path")
    if not _safe_relative_path(relative):
        failures.append(f"{path}: {name}.relative_path is required")
    digest = value.get("source_sha256", value.get("sha256"))
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        failures.append(f"{path}: {name}.source_sha256 is required")
    for field in ("current_git_blob", "head_git_blob"):
        if (
            not isinstance(value.get(field), str)
            or HEX40.fullmatch(str(value.get(field))) is None
        ):
            failures.append(f"{path}: {name}.{field} is required")
    if value.get("current_git_blob") != value.get("head_git_blob"):
        failures.append(f"{path}: {name} current and HEAD Git blobs must match")
    return failures


def _safe_relative_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not Path(value).is_absolute()
        and ".." not in Path(value).parts
    )


class DuplicateJsonKeyError(ValueError):
    """Raised for ambiguous machine-readable evidence."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def strict_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non_finite_json_constant:{value}")
        ),
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


def _plan_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["plan_sha256"] = "0" * 64
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _calibration_entries(payload: Mapping[str, Any], path: Path) -> dict[str, int]:
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"{path}: schema must be {CALIBRATION_SCHEMA}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: calibration entries are required")
    result: dict[str, int] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: entry {index} must be an object")
        key = item.get("key")
        count = item.get("loop_count")
        if not isinstance(key, str) or not key or "\n" in key:
            raise ValueError(f"{path}: entry {index} has an invalid key")
        count = _positive_int(count, f"{path}: entry {index} loop_count")
        if count > CALIBRATION_LOOP_COUNT_CEILING:
            raise ValueError(
                f"{path}: entry {index} loop_count exceeds calibration ceiling "
                f"{CALIBRATION_LOOP_COUNT_CEILING}"
            )
        if key in result:
            raise ValueError(f"{path}: duplicate calibration key {key!r}")
        result[key] = count
    declared_count = payload.get("entry_count")
    if declared_count != len(result):
        raise ValueError(f"{path}: entry_count does not match unique entries")
    return result


def _binding(
    payload: Mapping[str, Any],
    path: Path,
    *,
    plan_path: Path | None,
) -> dict[str, Any]:
    """Keep immutable source/workload identity without trusting it for timing."""

    resolved_path = path.resolve()
    relative_root = (
        plan_path.resolve().parent if plan_path is not None else resolved_path.parent
    )
    binding: dict[str, Any] = {
        # ``path`` remains the exact build-host identity used by the live
        # runner.  ``relative_path`` makes the same binding re-openable when a
        # complete evidence directory is moved for independent review.
        "path": str(resolved_path),
        "relative_path": Path(
            os.path.relpath(resolved_path, start=relative_root)
        ).as_posix(),
        "sha256": _sha256(path),
    }
    for key in (
        "backend",
        "variant",
        "source_commit",
        "product_source_commit",
        "harness_source_commit",
        "workload",
        "input_policy",
        "input_identity",
        "device",
        "binary",
        "payload",
        "wheel",
        "source_root",
        "product_source_root",
        "harness_source_root",
        "source_tree_state",
        "product_source_tree_state",
        "harness_source_tree_state",
        "source_blob",
        "harness_source_blob",
        "git",
        "git_identity",
    ):
        if key in payload:
            binding[key] = payload[key]
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        binding["provenance"] = dict(provenance)
    if "input_identity" in payload:
        binding["input_identity"] = payload["input_identity"]
    if "command_line" in payload:
        binding["command_line"] = payload["command_line"]
    return binding


def _identity_failures(payload: Mapping[str, Any], path: Path) -> list[str]:
    failures: list[str] = []
    source_commit = payload.get("source_commit")
    product_source_commit = payload.get("product_source_commit")
    harness_source_commit = payload.get("harness_source_commit")
    if (
        not isinstance(source_commit, str)
        or HEX40.fullmatch(source_commit) is None
        or product_source_commit != source_commit
    ):
        failures.append(
            f"{path}: matching full source_commit and product_source_commit are required"
        )
    if (
        not isinstance(harness_source_commit, str)
        or HEX40.fullmatch(harness_source_commit) is None
    ):
        failures.append(f"{path}: full harness_source_commit is required")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return [f"{path}: provenance is required"]
    binary = provenance.get("benchmark_binary", provenance.get("helper"))
    if not isinstance(binary, Mapping):
        failures.append(f"{path}: benchmark_binary identity is required")
    else:
        if not isinstance(binary.get("path"), str) or not binary.get("path"):
            failures.append(f"{path}: benchmark_binary path is required")
        if (
            not isinstance(binary.get("sha256"), str)
            or HEX64.fullmatch(binary.get("sha256", "")) is None
        ):
            failures.append(f"{path}: benchmark_binary sha256 is required")
    for name in ("payload", "wheel"):
        value = provenance.get(name)
        if value is None:
            continue
        if (
            not isinstance(value, Mapping)
            or HEX64.fullmatch(str(value.get("sha256", ""))) is None
        ):
            failures.append(f"{path}: {name} identity must contain sha256 or be null")
    input_identity = payload.get("input_identity", payload.get("dataset_identity"))
    if not isinstance(input_identity, Mapping) or not input_identity:
        failures.append(f"{path}: input_identity is required")
    command_line = payload.get("command_line")
    if not isinstance(command_line, list) or not command_line:
        failures.append(f"{path}: command_line is required")
    for name in ("source_root", "product_source_root", "harness_source_root"):
        value = payload.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            failures.append(f"{path}: absolute {name} is required")
    for name in (
        "source_tree_state",
        "product_source_tree_state",
        "harness_source_tree_state",
    ):
        value = payload.get(name)
        if (
            not isinstance(value, Mapping)
            or value.get("status") != "clean"
            or not isinstance(value.get("entries"), list)
        ):
            failures.append(f"{path}: clean {name} is required")
    failures.extend(_require_source_binding(payload, "source_blob", path))
    failures.extend(_require_source_binding(payload, "harness_source_blob", path))
    failures.extend(_require_git_identity(payload, path))
    return failures


def _scope(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    backend = payload.get("backend")
    workload = payload.get("workload")
    input_policy = payload.get("input_policy")
    evidence_lane = payload.get("evidence_lane")
    artifact_kind = payload.get("artifact_kind")
    device = payload.get("device", payload.get("device_identity"))
    scope_id = payload.get("scope_id")
    if not isinstance(backend, str) or not backend:
        raise ValueError(f"{path}: backend scope is required")
    if not isinstance(workload, Mapping) or not workload:
        raise ValueError(f"{path}: workload scope is required")
    for field in ("rows", "features", "candidates", "arity", "mi_bins", "top_k"):
        if (
            field not in workload
            or isinstance(workload[field], bool)
            or not isinstance(workload[field], int)
        ):
            raise ValueError(f"{path}: workload.{field} scope is required")
    if not isinstance(input_policy, str) or not input_policy:
        raise ValueError(f"{path}: input_policy scope is required")
    if not isinstance(evidence_lane, str) or not evidence_lane:
        raise ValueError(f"{path}: evidence_lane scope is required")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise ValueError(f"{path}: artifact_kind scope is required")
    if not isinstance(device, Mapping) or not device:
        raise ValueError(f"{path}: device scope is required")
    if not isinstance(scope_id, str) or not scope_id:
        raise ValueError(f"{path}: scope_id is required")
    return {
        "backend": backend,
        "workload": dict(workload),
        "input_policy": input_policy,
        "evidence_lane": evidence_lane,
        "artifact_kind": artifact_kind,
        "device": dict(device),
        "scope_id": scope_id,
    }


def make_plan(
    calibration_paths: Sequence[Path],
    *,
    headroom_factor: int = DEFAULT_HEADROOM_FACTOR,
    max_loop_count: int = DEFAULT_MAX_LOOP_COUNT,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    if len(calibration_paths) != 2:
        raise ValueError(
            "exactly baseline and candidate calibration artifacts are required"
        )
    headroom_factor = _positive_int(headroom_factor, "headroom_factor")
    max_loop_count = _positive_int(max_loop_count, "max_loop_count")
    if headroom_factor != DEFAULT_HEADROOM_FACTOR:
        raise ValueError(
            f"headroom_factor must equal the fixed policy value "
            f"{DEFAULT_HEADROOM_FACTOR}"
        )
    if max_loop_count != DEFAULT_MAX_LOOP_COUNT:
        raise ValueError(
            f"max_loop_count must equal the fixed plan ceiling "
            f"{DEFAULT_MAX_LOOP_COUNT}"
        )
    combined: dict[str, int] = {}
    bindings: list[dict[str, Any]] = []
    scopes: list[dict[str, Any]] = []
    variants: list[str] = []
    source_commits: list[str] = []
    harness_source_commits: list[str] = []
    key_sets: list[set[str]] = []
    for path in calibration_paths:
        payload = strict_load(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: calibration root must be an object")
        identity_failures = _identity_failures(payload, path)
        if identity_failures:
            raise ValueError("; ".join(identity_failures))
        entries = _calibration_entries(payload, path)
        bindings.append(_binding(payload, path, plan_path=plan_path))
        scopes.append(_scope(payload, path))
        variant = payload.get("variant")
        source_commit = payload.get("source_commit")
        if not isinstance(variant, str) or not variant:
            raise ValueError(f"{path}: variant is required")
        if not isinstance(source_commit, str) or HEX40.fullmatch(source_commit) is None:
            raise ValueError(f"{path}: full source_commit is required")
        variants.append(variant)
        source_commits.append(source_commit)
        harness_source_commits.append(str(payload["harness_source_commit"]))
        key_sets.append(set(entries))
        for key, count in entries.items():
            combined[key] = max(combined.get(key, 0), count)
    if set(variants) != {"baseline", "candidate"} or len(variants) != 2:
        raise ValueError("calibration artifacts must be exactly baseline and candidate")
    if len(set(source_commits)) != 2:
        raise ValueError("baseline and candidate product commits must differ")
    if len(set(harness_source_commits)) != 1:
        raise ValueError(
            "baseline and candidate calibration harness commits must match"
        )
    if key_sets[0] != key_sets[1]:
        raise ValueError(
            "baseline and candidate calibration key sets must match exactly"
        )
    if scopes[0] != scopes[1]:
        raise ValueError(
            "baseline and candidate calibration scopes must match exactly: "
            + json.dumps(scopes, sort_keys=True)
        )

    entries: list[dict[str, Any]] = []
    for key in sorted(combined):
        observed = combined[key]
        # The helpers use powers of two.  A fixed factor of two is deliberately
        # explicit and deterministic; it gives the recorded region one
        # calibration guard band without allowing per-variant adaptation.
        planned = observed * headroom_factor
        if planned > max_loop_count:
            raise ValueError(
                f"loop-plan cap exceeded for {key!r}: {planned} > {max_loop_count}"
            )
        entries.append({"key": key, "loop_count": planned})
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "policy": "max_calibration_count_times_fixed_headroom_factor",
        "headroom_factor": headroom_factor,
        "max_loop_count": max_loop_count,
        "source_count": 2,
        "scope": scopes[0],
        "bindings": sorted(bindings, key=lambda item: str(item["path"])),
        "variants": sorted(variants),
        "source_commits": sorted(source_commits),
        "entry_count": len(entries),
        "entries": entries,
        "plan_sha256": "0" * 64,
    }
    payload["plan_sha256"] = _plan_digest(payload)
    return payload


def validate_plan(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != SCHEMA:
        failures.append("schema_mismatch")
    if payload.get("version") != 1:
        failures.append("version_mismatch")
    if payload.get("source_count") != 2:
        failures.append("source_count_must_be_two")
    if payload.get("variants") != ["baseline", "candidate"]:
        failures.append("baseline_and_candidate_variants_required")
    commits = payload.get("source_commits")
    if (
        not isinstance(commits, list)
        or len(commits) != 2
        or any(
            not isinstance(item, str) or HEX40.fullmatch(item) is None
            for item in commits
        )
        or len(set(commits)) != 2
    ):
        failures.append("distinct_full_source_commits_required")
    if payload.get("policy") != "max_calibration_count_times_fixed_headroom_factor":
        failures.append("policy_mismatch")
    factor = payload.get("headroom_factor")
    cap = payload.get("max_loop_count")
    if factor != DEFAULT_HEADROOM_FACTOR:
        failures.append("headroom_factor_invalid")
    if cap != DEFAULT_MAX_LOOP_COUNT:
        failures.append("max_loop_count_invalid")
    entries = payload.get("entries")
    keys: set[str] = set()
    if not isinstance(entries, list) or not entries:
        failures.append("entries_required")
        entries = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            failures.append(f"entry_{index}_must_be_object")
            continue
        key = item.get("key")
        count = item.get("loop_count")
        if not isinstance(key, str) or not key:
            failures.append(f"entry_{index}_key_invalid")
        elif key in keys:
            failures.append(f"entry_{index}_duplicate_key")
        else:
            keys.add(key)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or (isinstance(cap, int) and count > cap)
        ):
            failures.append(f"entry_{index}_loop_count_invalid")
    if payload.get("entry_count") != len(entries):
        failures.append("entry_count_mismatch")
    if not isinstance(payload.get("bindings"), list) or not payload["bindings"]:
        failures.append("bindings_required")
    elif len(payload["bindings"]) != 2:
        failures.append("binding_count_must_be_two")
    else:
        observed_variants: list[str] = []
        observed_commits: list[str] = []
        observed_harness_commits: list[str] = []
        for index, binding in enumerate(payload["bindings"]):
            if not isinstance(binding, Mapping):
                failures.append(f"binding_{index}_must_be_object")
                continue
            path = binding.get("path")
            relative_path = binding.get("relative_path")
            digest = binding.get("sha256")
            variant = binding.get("variant")
            commit = binding.get("source_commit")
            product_commit = binding.get("product_source_commit")
            harness_commit = binding.get("harness_source_commit")
            if not isinstance(path, str) or not path:
                failures.append(f"binding_{index}_path_required")
            if not _safe_relative_path(relative_path):
                failures.append(f"binding_{index}_relative_path_required")
            if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
                failures.append(f"binding_{index}_sha256_required")
            if not isinstance(variant, str) or not variant:
                failures.append(f"binding_{index}_variant_required")
            else:
                observed_variants.append(variant)
            if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
                failures.append(f"binding_{index}_source_commit_required")
            else:
                observed_commits.append(commit)
            if product_commit != commit:
                failures.append(f"binding_{index}_product_source_commit_mismatch")
            if (
                not isinstance(harness_commit, str)
                or HEX40.fullmatch(harness_commit) is None
            ):
                failures.append(f"binding_{index}_harness_source_commit_required")
            else:
                observed_harness_commits.append(harness_commit)
        if sorted(observed_variants) != ["baseline", "candidate"]:
            failures.append("binding_variants_mismatch")
        if len(observed_commits) != 2 or len(set(observed_commits)) != 2:
            failures.append("binding_source_commits_must_differ")
        if (
            len(observed_harness_commits) != 2
            or len(set(observed_harness_commits)) != 1
        ):
            failures.append("binding_harness_source_commits_must_match")
        if isinstance(commits, list) and sorted(observed_commits) != sorted(commits):
            failures.append("binding_source_commits_root_mismatch")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        failures.append("scope_required")
    else:
        for field in (
            "backend",
            "workload",
            "input_policy",
            "evidence_lane",
            "artifact_kind",
            "device",
            "scope_id",
        ):
            if field not in scope or not scope[field]:
                failures.append(f"scope_{field}_required")
        workload = scope.get("workload")
        if not isinstance(workload, Mapping):
            failures.append("scope_workload_required")
        else:
            for field in (
                "rows",
                "features",
                "candidates",
                "arity",
                "mi_bins",
                "top_k",
            ):
                if (
                    field not in workload
                    or isinstance(workload[field], bool)
                    or not isinstance(workload[field], int)
                ):
                    failures.append(f"scope_workload_{field}_required")
    digest = payload.get("plan_sha256")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        failures.append("plan_sha256_required")
    elif digest != _plan_digest(payload):
        failures.append("plan_sha256_mismatch")
    return failures


def write_plan(output: Path, payload: Mapping[str, Any]) -> None:
    failures = validate_plan(payload)
    if failures:
        raise ValueError("invalid loop plan: " + ", ".join(failures))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_canonical_json(payload), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", action="append", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--headroom-factor", type=int, default=DEFAULT_HEADROOM_FACTOR)
    parser.add_argument("--max-loop-count", type=int, default=DEFAULT_MAX_LOOP_COUNT)
    args = parser.parse_args(argv)
    calibration_paths = list(args.calibration or ())
    if args.baseline is not None or args.candidate is not None:
        if args.baseline is None or args.candidate is None or calibration_paths:
            parser.error("use either --calibration twice or --baseline/--candidate")
        calibration_paths = [args.baseline, args.candidate]
    if len(calibration_paths) != 2:
        parser.error("exactly two calibration artifacts are required")
    payload = make_plan(
        calibration_paths,
        headroom_factor=args.headroom_factor,
        max_loop_count=args.max_loop_count,
        plan_path=args.output,
    )
    write_plan(args.output, payload)
    print(payload["plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
