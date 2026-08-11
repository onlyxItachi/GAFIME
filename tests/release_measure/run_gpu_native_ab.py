#!/usr/bin/env python3
"""Collect lane-isolated CUDA/ROCm native A/B evidence.

This runner is deliberately an orchestration layer.  It never publishes or
mutates a package/release; it starts one helper process per calibration and
recorded cell, creates immutable loop plans, and writes hash-bound native
evidence manifests for ``perf_13_precision_profiles.py``.

The recorded matrix is fixed at 24 helper processes per backend::

    3 evidence lanes * 2 input policies * 2 A/B blocks * 2 variants

The calibration matrix is 12 additional helper processes.  Every recorded
process consumes the same lane/policy loop plan built from both variants.
CUDA and ROCm require compile-time lane-specific helpers for all three lanes.
The direct benchmark executable embeds product device code; avoiding a direct
call at runtime is not enough to exclude fatbin registration/module state.
The canonical and host helpers are separate common-harness binaries with no
product translation unit, and every invocation remains a fresh process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Mapping, Sequence


LANES = (
    "canonical_payload_api",
    "supplemental_internal_kernel",
    "supplemental_host_phase",
)
INPUT_POLICIES = ("common-f64", "native")
PROFILES = ("fp32", "mixed", "fp64")
LANE_ISOLATION = "fresh_helper_process_per_variant_trial_and_lane"
PROCESS_ISOLATION = "fresh_helper_process_per_variant_trial"
CALIBRATION_PROCESS_ISOLATION = "fresh_helper_process_per_variant_calibration_and_lane"
RUNNER_SCHEMA = "gafime.gpu-native-ab-runner.v1"
MANIFEST_SCHEMA = "gafime.precision-profile-native-evidence.v1"
CALIBRATION_SCHEMA = "gafime.native-loop-calibration.v1"
LOOP_PLAN_SCHEMA = "gafime.native-loop-plan.v1"
BACKENDS = {"cuda", "rocm"}
ARTIFACT_KINDS = {"cuda": "cuda_events", "rocm": "rocm_events"}
ARTIFACT_SCHEMAS = {
    "cuda": "gafime.cuda.native_timing.v2",
    "rocm": "gafime.rocm.native_timing.v2",
}
EXPECTED_RECORDED_PROCESSES = len(LANES) * len(INPUT_POLICIES) * 2 * 2
EXPECTED_CALIBRATION_PROCESSES = len(LANES) * len(INPUT_POLICIES) * 2

# Git identity is part of the evidence boundary.  Never resolve it through
# PATH: a repository-local or temporary ``git`` wrapper must not be able to
# manufacture a clean tree/commit for a release measurement.
TRUSTED_GIT_CANDIDATES = (Path("/usr/bin/git"), Path("/bin/git"))

ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "GAFIME_CUDA_V1_LIB",
    "GAFIME_ROCM_V1_LIB",
    "GAFIME_NATIVE_AFFINITY",
    "GAFIME_NATIVE_RUNNER_INVOCATION_ID",
    "GAFIME_NATIVE_RUNNER_PID",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCM_PATH",
    "HIP_PATH",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)


def _trusted_git_executable() -> Path:
    for candidate in TRUSTED_GIT_CANDIDATES:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise RunnerError("no trusted absolute Git executable is available")


def _git_environment() -> tuple[dict[str, str], list[str]]:
    removed = sorted(key for key in os.environ if key.startswith("GIT_"))
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    # Disable system/global config without inheriting any path redirection.
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    return environment, removed


class RunnerError(RuntimeError):
    """A fail-closed orchestration or evidence error."""


def _canonical_json(value: object) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_load(path: Path) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RunnerError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnerError(f"cannot read strict JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload), encoding="utf-8")


def _file_identity(path: Path, label: str, *, dry_run: bool) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        if dry_run:
            return {"path": str(path), "size_bytes": 0, "sha256": None, "dry_run": True}
        raise RunnerError(f"{label} must be an existing regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git(root: Path, *args: str) -> str:
    executable = _trusted_git_executable()
    environment, _ = _git_environment()
    try:
        result = subprocess.run(
            [str(executable), "-C", str(root), *args],
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"git identity query failed for {root}: {exc}") from exc
    return result.stdout.strip()


def _source_identity(root: Path, label: str, *, dry_run: bool) -> dict[str, object]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        if dry_run:
            return {"root": str(root), "status": "dry_run_unverified"}
        raise RunnerError(f"{label} must be an existing directory: {root}")
    try:
        executable = _trusted_git_executable()
        _, removed_git_environment = _git_environment()
        reported_top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
        if reported_top_level != root:
            raise RunnerError(
                f"{label} Git top-level does not physically match source root: {root}"
            )
        actual_git_entry = root / ".git"
        if actual_git_entry.is_dir():
            expected_git_dir = actual_git_entry.resolve()
            expected_common_dir = expected_git_dir
        elif actual_git_entry.is_file():
            marker = actual_git_entry.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                raise RunnerError(f"{label} .git file has no gitdir target: {root}")
            expected_git_dir = Path(marker[len("gitdir:") :].strip())
            if not expected_git_dir.is_absolute():
                expected_git_dir = (root / expected_git_dir).resolve()
            else:
                expected_git_dir = expected_git_dir.resolve()
            expected_common_dir = (
                expected_git_dir.parent.parent
                if expected_git_dir.parent.name == "worktrees"
                else expected_git_dir.parent
            ).resolve()
        else:
            raise RunnerError(f"{label} source root has no physical .git entry: {root}")
        reported_git_dir = Path(_git(root, "rev-parse", "--git-dir"))
        if not reported_git_dir.is_absolute():
            reported_git_dir = (root / reported_git_dir).resolve()
        else:
            reported_git_dir = reported_git_dir.resolve()
        reported_common_dir = Path(_git(root, "rev-parse", "--git-common-dir"))
        if not reported_common_dir.is_absolute():
            reported_common_dir = (root / reported_common_dir).resolve()
        else:
            reported_common_dir = reported_common_dir.resolve()
        if reported_git_dir != expected_git_dir:
            raise RunnerError(
                f"{label} Git dir does not match physical .git entry: {root}"
            )
        if (
            reported_common_dir != expected_common_dir
            or not reported_common_dir.is_dir()
        ):
            raise RunnerError(
                f"{label} Git common dir does not belong to source root: {root}"
            )
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
        commit = _git(root, "rev-parse", "HEAD")
        tree = _git(root, "rev-parse", "HEAD^{tree}")
        version = _git(root, "--version")
    except RunnerError:
        if dry_run:
            return {"root": str(root), "status": "dry_run_unverified"}
        raise
    if status:
        raise RunnerError(f"{label} must be a clean Git worktree: {root}")
    if len(commit) != 40 or len(tree) != 40:
        raise RunnerError(f"{label} Git identity is not full length: {root}")
    return {
        "root": str(root),
        "status": "clean",
        "commit": commit,
        "tree": tree,
        "git": {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "version": version,
            "git_dir": str(reported_git_dir),
            "git_common_dir": str(reported_common_dir),
            "removed_environment": removed_git_environment,
        },
    }


def _affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(int(value) for value in os.sched_getaffinity(0))
    except OSError:
        return None


def _environment_snapshot(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        key: str(environment[key]) for key in ENVIRONMENT_KEYS if key in environment
    }


def _parse_lane_bindings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise RunnerError(f"lane helper binding must be LANE=PATH, got {raw!r}")
        lane, raw_path = raw.split("=", 1)
        if lane not in LANES or not raw_path:
            raise RunnerError(f"unknown or empty lane helper binding: {raw!r}")
        if lane in result:
            raise RunnerError(f"duplicate lane helper binding: {lane}")
        result[lane] = Path(raw_path).expanduser().resolve()
    return result


@dataclass(frozen=True)
class Variant:
    name: str
    helper: Path
    payload: Path
    wheel: Path
    source_root: Path
    canonical_evidence: Path
    lane_helpers: Mapping[str, Path]

    def helper_for(self, backend: str, lane: str, *, dry_run: bool) -> Path:
        if backend not in BACKENDS:
            raise RunnerError(f"unsupported backend: {backend}")
        if lane not in LANES:
            raise RunnerError(f"unsupported evidence lane: {lane}")
        direct = self.helper.expanduser().resolve()
        if lane == "supplemental_internal_kernel":
            if (
                lane in self.lane_helpers
                and self.lane_helpers[lane].resolve() != direct
            ):
                raise RunnerError(
                    f"{backend.upper()} direct lane binding for {self.name} must match "
                    f"--{self.name}-helper"
                )
            selected = direct
        else:
            selected = self.lane_helpers.get(lane)
            if selected is None:
                raise RunnerError(
                    f"{backend.upper()} {lane} requires a lane-specific helper binary for {self.name}; "
                    "the direct-kernel helper cannot be reused"
                )
            selected = selected.expanduser().resolve()
        if not dry_run:
            # Check all three identities together on every selection.  This
            # prevents a helper accepted for one lane from being silently
            # reused by another lane later in the matrix.
            lane_paths = {
                "supplemental_internal_kernel": direct,
                "canonical_payload_api": self.lane_helpers.get(
                    "canonical_payload_api", Path()
                )
                .expanduser()
                .resolve(),
                "supplemental_host_phase": self.lane_helpers.get(
                    "supplemental_host_phase", Path()
                )
                .expanduser()
                .resolve(),
            }
            if all(path.is_file() for path in lane_paths.values()):
                hashes = {
                    lane_name: _sha256_file(path)
                    for lane_name, path in lane_paths.items()
                }
                if len(set(hashes.values())) != len(hashes):
                    raise RunnerError(
                        f"{backend.upper()} helpers for {self.name} must be byte-distinct across "
                        "supplemental_internal_kernel, canonical_payload_api, and "
                        "supplemental_host_phase"
                    )
        return selected


@dataclass(frozen=True)
class RunnerConfig:
    backend: str
    workload: str
    input_policies: tuple[str, ...]
    baseline: Variant
    candidate: Variant
    harness_source_root: Path
    output_dir: Path
    loop_plan_script: Path
    rows: int = 4096
    features: int = 8
    candidates: int = 8
    arity: int = 1
    mi_bins: int = 32
    top_k: int = 2
    warmups: int = 10
    repeats: int = 30
    order_repetitions: int = 30
    seed: int = 20260809
    dataset_seed: int = 0x524F434D31312D31
    device: int = 0
    affinity: tuple[int, ...] | None = None


def _split_policies(value: str | Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    source = [value] if isinstance(value, str) else list(value)
    for item in source:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    if not values or set(values) != set(INPUT_POLICIES) or len(values) != 2:
        raise RunnerError(
            f"input policy must contain exactly common-f64,native; got {values!r}"
        )
    return tuple(values)


def _parse_affinity(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit():
            raise RunnerError("affinity must be a comma-separated list of CPU numbers")
        result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise RunnerError("affinity must contain distinct CPU numbers")
    return tuple(sorted(result))


def _variant_sequence(block: int) -> tuple[str, str]:
    if block == 0:
        return ("baseline", "candidate")
    if block == 1:
        return ("candidate", "baseline")
    raise RunnerError(f"unsupported A/B block {block}")


def _validate_config(config: RunnerConfig) -> None:
    if config.backend not in BACKENDS:
        raise RunnerError(f"unsupported backend: {config.backend}")
    if not config.workload:
        raise RunnerError("workload must not be empty")
    if (
        set(config.input_policies) != set(INPUT_POLICIES)
        or len(config.input_policies) != 2
    ):
        raise RunnerError("both common-f64 and native input policies are required")
    positive = {
        "rows": config.rows,
        "features": config.features,
        "candidates": config.candidates,
        "mi_bins": config.mi_bins,
        "top_k": config.top_k,
        "warmups": config.warmups,
        "repeats": config.repeats,
        "order_repetitions": config.order_repetitions,
    }
    for name, value in positive.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RunnerError(f"{name} must be a positive integer")
    if config.arity < 1 or config.arity > 5 or config.arity > config.features:
        raise RunnerError("arity must be in 1..5 and no greater than features")
    if config.mi_bins not in {2, 4, 8, 12, 16, 24, 32, 48, 64, 96}:
        raise RunnerError("mi-bins must be one of 2,4,8,12,16,24,32,48,64,96")
    if config.top_k > config.candidates:
        raise RunnerError("top-k cannot exceed candidates")
    if config.warmups < 10 or config.repeats < 30 or config.order_repetitions < 30:
        raise RunnerError(
            "warmups must be at least 10, repeats at least 30, and order-repetitions at least 30"
        )
    if (
        isinstance(config.device, bool)
        or not isinstance(config.device, int)
        or config.device < 0
    ):
        raise RunnerError("device must be a non-negative integer")
    for name, value in (("seed", config.seed), ("dataset-seed", config.dataset_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError(f"{name} must be a non-negative integer")
    if config.affinity is None:
        raise RunnerError(
            "an explicit single-CPU --affinity is required for claim-bearing native evidence"
        )
    if len(config.affinity) != 1 or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
        for cpu in config.affinity
    ):
        raise RunnerError("affinity must contain exactly one non-negative CPU number")
    available = _affinity()
    if available is not None and not set(config.affinity) <= set(available):
        raise RunnerError(
            f"requested affinity {list(config.affinity)} is outside the current CPU mask {available}"
        )


def _reject_stale_outputs(paths: Sequence[Path], *, dry_run: bool) -> None:
    if dry_run:
        return
    stale = sorted({str(path) for path in paths if path.exists() or path.is_symlink()})
    if stale:
        raise RunnerError(
            "output paths already exist; use a new output directory to prevent stale evidence reuse: "
            + ", ".join(stale)
        )


def _helper_args(
    config: RunnerConfig,
    variant: Variant,
    helper: Path,
    *,
    lane: str,
    input_policy: str,
    output: Path,
    calibration_only: bool,
    plan: Path | None = None,
    variant_name: str | None = None,
    block: int | None = None,
    sequence: Sequence[str] = (),
) -> list[str]:
    args = [
        str(helper),
        "--payload",
        str(variant.payload),
        "--wheel",
        str(variant.wheel),
        "--source-root",
        str(variant.source_root),
        "--harness-source-root",
        str(config.harness_source_root),
        "--workload",
        config.workload,
        "--input-policy",
        input_policy,
        "--evidence-lane",
        lane,
        "--artifact-kind",
        ARTIFACT_KINDS[config.backend],
        "--json",
        str(output),
        "--rows",
        str(config.rows),
        "--features",
        str(config.features),
        "--candidates",
        str(config.candidates),
        "--arity",
        str(config.arity),
        "--mi-bins",
        str(config.mi_bins),
        "--top-k",
        str(config.top_k),
        "--warmups",
        str(config.warmups),
        "--repeats",
        str(config.repeats),
        "--order-repetitions",
        str(config.order_repetitions),
        "--device",
        str(config.device),
    ]
    if config.backend == "cuda":
        args.extend(
            (
                "--profiles",
                ",".join(PROFILES),
                "--seed",
                str(config.seed),
                "--csv",
                str(output.with_suffix(".csv")),
            )
        )
    else:
        args.extend(
            (
                "--order-seed",
                str(config.seed),
                "--dataset-seed",
                str(config.dataset_seed),
            )
        )
    args.extend(("--canonical-evidence", str(variant.canonical_evidence)))
    if calibration_only:
        args.append("--calibration-only")
        if variant_name is not None:
            # Calibration artifacts must carry the variant identity so the
            # immutable plan can bind distinct baseline/candidate commits.
            # The helper accepts this without an A/B block in calibration mode.
            args.extend(("--variant", variant_name))
    else:
        if plan is None:
            raise RunnerError(
                "recorded helper invocation requires an immutable loop plan"
            )
        args.extend(("--loop-plan", str(plan)))
        if variant_name is not None:
            if (
                block is None
                or tuple(sequence)
                not in {
                    ("baseline", "candidate"),
                    ("candidate", "baseline"),
                }
                or variant_name not in sequence
            ):
                raise RunnerError("recorded helper schedule metadata is incomplete")
            args.extend(
                (
                    "--variant",
                    variant_name,
                    "--ab-block",
                    str(block),
                    "--variant-sequence",
                    ",".join(sequence),
                )
            )
    return args


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    dry_run: bool,
    process_affinity: Sequence[int] | None = None,
) -> dict[str, object]:
    if dry_run:
        return {"returncode": 0, "stdout": "", "stderr": "", "dry_run": True}
    try:
        kwargs: dict[str, object] = {}
        if process_affinity is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RunnerError(
                    "requested process affinity cannot be enforced on this platform"
                )
            requested = {int(value) for value in process_affinity}
            if not requested:
                raise RunnerError("requested process affinity must not be empty")

            def pin_child() -> None:
                os.sched_setaffinity(0, requested)  # type: ignore[attr-defined]

            # Each helper is a fresh process.  Pinning in the child avoids
            # mutating the runner's own affinity while making the measured
            # affinity an enforced, rather than merely observed, property.
            kwargs["preexec_fn"] = pin_child
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        stdout, stderr = process.communicate()
    except OSError as exc:
        raise RunnerError(f"failed to start fresh helper process: {exc}") from exc
    if process.returncode != 0:
        raise RunnerError(
            "fresh helper process failed with exit code "
            f"{process.returncode}: {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "child_pid": process.pid,
        "dry_run": False,
    }


def _payload_environment(payload: Mapping[str, object], path: Path) -> dict[str, str]:
    raw = payload.get("environment")
    if isinstance(raw, Mapping):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        values: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                raise RunnerError(f"evidence environment is malformed: {path}")
            key, value = item.split("=", 1)
            if not key or key in values:
                raise RunnerError(f"evidence environment is ambiguous: {path}")
            values[key] = value
        return values
    raise RunnerError(f"evidence environment is missing: {path}")


def _expected_lane_contract(lane: str) -> dict[str, object]:
    if lane not in LANES:
        raise RunnerError(f"unsupported evidence lane: {lane}")
    canonical = lane == "canonical_payload_api"
    return {
        "execution_mode": "canonical_payload" if canonical else lane,
        "payload_loaded": canonical,
        "payload_not_loaded": not canonical,
        "payload_execution_mode": "canonical_payload"
        if canonical
        else "payload_not_loaded",
    }


def _validate_lane_contract(
    payload: Mapping[str, object], path: Path, *, lane: str, artifact: str
) -> None:
    for field, expected in _expected_lane_contract(lane).items():
        if payload.get(field) != expected:
            raise RunnerError(
                f"{artifact} {field} mismatch in {path}: "
                f"got {payload.get(field)!r}, expected {expected!r}"
            )


def _validate_process_attestation(
    payload: Mapping[str, object],
    path: Path,
    *,
    expected_environment: Mapping[str, str],
    expected_affinity: Sequence[int],
    expected_child_pid: int,
) -> None:
    environment = _payload_environment(payload, path)
    for key in (
        "GAFIME_WHEEL_PATH",
        "GAFIME_CUDA_V1_LIB",
        "GAFIME_ROCM_V1_LIB",
        "GAFIME_NATIVE_RUNNER_INVOCATION_ID",
        "GAFIME_NATIVE_RUNNER_PID",
    ):
        if (
            key in expected_environment
            and environment.get(key) != expected_environment[key]
        ):
            raise RunnerError(f"evidence environment {key} mismatch: {path}")
    runner_pid = payload.get("runner_pid")
    invocation_id = payload.get("runner_invocation_id")
    child_pid = payload.get("process_id")
    if str(runner_pid) != expected_environment.get("GAFIME_NATIVE_RUNNER_PID"):
        raise RunnerError(f"runner PID attestation mismatch: {path}")
    if invocation_id != expected_environment.get("GAFIME_NATIVE_RUNNER_INVOCATION_ID"):
        raise RunnerError(f"runner invocation attestation mismatch: {path}")
    if (
        isinstance(child_pid, bool)
        or not isinstance(child_pid, int)
        or child_pid != expected_child_pid
        or child_pid == runner_pid
    ):
        raise RunnerError(f"fresh child-process attestation mismatch: {path}")
    observed_affinity = payload.get("process_affinity", payload.get("affinity"))
    if isinstance(observed_affinity, Mapping):
        current_cpu = observed_affinity.get("current_cpu")
        observed_affinity = observed_affinity.get(
            "cpus", observed_affinity.get("allowed_cpus")
        )
    else:
        current_cpu = payload.get("current_cpu")
    if not isinstance(observed_affinity, list) or sorted(observed_affinity) != sorted(
        expected_affinity
    ):
        raise RunnerError(f"recorded process affinity mismatch: {path}")
    if current_cpu is not None and current_cpu not in expected_affinity:
        raise RunnerError(
            f"recorded current CPU is outside the pinned affinity: {path}"
        )


def _validate_calibration(
    path: Path,
    *,
    config: RunnerConfig,
    variant_spec: Variant,
    expected_source_commit: str,
    expected_harness_commit: str,
    expected_command: Sequence[str],
    expected_environment: Mapping[str, str],
    expected_affinity: Sequence[int],
    expected_child_pid: int,
    lane: str,
    policy: str,
    variant: str,
) -> dict[str, object]:
    payload = _strict_load(path)
    if not isinstance(payload, Mapping):
        raise RunnerError(f"calibration output must be an object: {path}")
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise RunnerError(f"calibration schema mismatch: {path}")
    if payload.get("status") != "calibration_only":
        raise RunnerError(f"calibration output is not calibration-only: {path}")
    if payload.get("backend") != config.backend:
        raise RunnerError(f"calibration backend mismatch: {path}")
    if payload.get("evidence_lane") != lane:
        raise RunnerError(f"calibration lane mismatch: {path}")
    if payload.get("input_policy") != policy:
        raise RunnerError(f"calibration input-policy mismatch: {path}")
    if payload.get("variant") != variant:
        raise RunnerError(f"calibration variant mismatch: {path}")
    if payload.get("artifact_kind") != ARTIFACT_KINDS[config.backend]:
        raise RunnerError(f"calibration artifact-kind mismatch: {path}")
    expected_load_status = (
        "loaded_canonical_lane_only"
        if lane == "canonical_payload_api"
        else "not_loaded_by_lane_contract"
    )
    if payload.get("payload_load_status") != expected_load_status:
        raise RunnerError(f"calibration payload-load status mismatch: {path}")
    expected_absence_attested = lane != "canonical_payload_api"
    if payload.get("payload_absence_attested") is not expected_absence_attested:
        raise RunnerError(f"calibration payload-absence attestation mismatch: {path}")
    _validate_lane_contract(payload, path, lane=lane, artifact="calibration")
    if payload.get("source_commit") != expected_source_commit:
        raise RunnerError(f"calibration source commit mismatch: {path}")
    if payload.get("product_source_commit") != expected_source_commit:
        raise RunnerError(f"calibration product source commit mismatch: {path}")
    if payload.get("harness_source_commit") != expected_harness_commit:
        raise RunnerError(f"calibration harness source commit mismatch: {path}")
    if payload.get("command_line") != list(expected_command):
        raise RunnerError(f"calibration command line mismatch: {path}")
    workload = payload.get("workload")
    expected_workload = {
        "name": config.workload,
        "rows": config.rows,
        "features": config.features,
        "candidates": config.candidates,
        "arity": config.arity,
        "mi_bins": config.mi_bins,
        "top_k": config.top_k,
    }
    if not isinstance(workload, Mapping) or any(
        workload.get(key) != value for key, value in expected_workload.items()
    ):
        raise RunnerError(f"calibration workload scope mismatch: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RunnerError(f"calibration provenance is missing: {path}")
    expected_files = {
        "benchmark_binary": Path(expected_command[0]),
        "payload": variant_spec.payload,
        "wheel": variant_spec.wheel,
    }
    for name, expected_path in expected_files.items():
        identity = provenance.get(name)
        if not isinstance(identity, Mapping):
            raise RunnerError(f"calibration provenance.{name} is missing: {path}")
        if Path(str(identity.get("path"))).resolve() != expected_path.resolve():
            raise RunnerError(f"calibration provenance.{name} path mismatch: {path}")
        if identity.get("sha256") != _sha256_file(expected_path):
            raise RunnerError(f"calibration provenance.{name} hash mismatch: {path}")
    _validate_process_attestation(
        payload,
        path,
        expected_environment=expected_environment,
        expected_affinity=expected_affinity,
        expected_child_pid=expected_child_pid,
    )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RunnerError(f"calibration entries are missing: {path}")
    keys: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            raise RunnerError(f"calibration entry {index} is malformed: {path}")
        key = str(item["key"])
        count = item.get("loop_count")
        if (
            not key
            or key in keys
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise RunnerError(f"calibration entry {index} is invalid: {path}")
        keys.add(key)
    if payload.get("entry_count") != len(entries):
        raise RunnerError(f"calibration entry_count mismatch: {path}")
    return dict(payload)


def _plan_digest(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned["plan_sha256"] = "0" * 64
    return _sha256_bytes(_canonical_json(unsigned).encode("utf-8"))


def _calibration_entry_map(path: Path, payload: Mapping[str, object]) -> dict[str, int]:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RunnerError(f"calibration entries are missing: {path}")
    result: dict[str, int] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise RunnerError(f"calibration entry {index} is malformed: {path}")
        key = item.get("key")
        count = item.get("loop_count")
        if (
            not isinstance(key, str)
            or not key
            or key in result
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise RunnerError(f"calibration entry {index} is invalid: {path}")
        result[key] = count
    if payload.get("entry_count") != len(result):
        raise RunnerError(f"calibration entry_count mismatch: {path}")
    return result


def _calibration_binding_view(payload: Mapping[str, object]) -> dict[str, object]:
    view: dict[str, object] = {}
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
            view[key] = payload[key]
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        view["provenance"] = dict(provenance)
    if "input_identity" in payload:
        view["input_identity"] = payload["input_identity"]
    if "command_line" in payload:
        view["command_line"] = payload["command_line"]
    return view


def _calibration_scope_view(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "backend": payload.get("backend"),
        "workload": payload.get("workload"),
        "input_policy": payload.get("input_policy"),
        "evidence_lane": payload.get("evidence_lane"),
        "artifact_kind": payload.get("artifact_kind"),
        "device": payload.get("device"),
        "scope_id": payload.get("scope_id"),
    }


def _validate_calibration_provenance(path: Path, payload: Mapping[str, object]) -> None:
    for name in ("source_root", "product_source_root", "harness_source_root"):
        if (
            not isinstance(payload.get(name), str)
            or not Path(str(payload[name])).is_absolute()
        ):
            raise RunnerError(f"calibration {name} is missing: {path}")
    for name in (
        "source_tree_state",
        "product_source_tree_state",
        "harness_source_tree_state",
    ):
        state = payload.get(name)
        if (
            not isinstance(state, Mapping)
            or state.get("status") != "clean"
            or not isinstance(state.get("entries"), list)
        ):
            raise RunnerError(f"calibration {name} is not clean: {path}")
    for name in ("source_blob", "harness_source_blob"):
        blob = payload.get(name)
        digest = (
            blob.get("source_sha256", blob.get("sha256"))
            if isinstance(blob, Mapping)
            else None
        )
        if (
            not isinstance(blob, Mapping)
            or not isinstance(blob.get("relative_path"), str)
            or not blob.get("relative_path")
            or ".." in Path(str(blob.get("relative_path"))).parts
            or Path(str(blob.get("relative_path"))).is_absolute()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or not isinstance(blob.get("current_git_blob"), str)
            or len(str(blob.get("current_git_blob"))) != 40
            or not isinstance(blob.get("head_git_blob"), str)
            or len(str(blob.get("head_git_blob"))) != 40
            or blob.get("current_git_blob") != blob.get("head_git_blob")
        ):
            raise RunnerError(
                f"calibration {name} tracked identity is malformed: {path}"
            )
    git = payload.get("git", payload.get("git_identity"))
    if not isinstance(git, Mapping):
        raise RunnerError(f"calibration trusted Git identity is missing: {path}")
    executable = _trusted_git_executable()
    if Path(str(git.get("path"))).expanduser().resolve() != executable:
        raise RunnerError(f"calibration Git executable path mismatch: {path}")
    if git.get("sha256") != _sha256_file(executable):
        raise RunnerError(f"calibration Git executable hash mismatch: {path}")
    if not isinstance(git.get("version"), str) or not git.get("version"):
        raise RunnerError(f"calibration Git executable version is missing: {path}")
    removed = git.get("removed_environment")
    if not isinstance(removed, list) or any(
        not isinstance(name, str) or not name.startswith("GIT_") for name in removed
    ):
        raise RunnerError(
            f"calibration Git environment scrub attestation is missing: {path}"
        )
    for name in ("git_dir", "git_common_dir"):
        if not isinstance(git.get(name), str) or not Path(str(git[name])).is_absolute():
            raise RunnerError(f"calibration {name} is missing: {path}")


def _reauthenticate_plan_calibrations(
    plan_path: Path,
    plan: Mapping[str, object],
    calibration_paths: Sequence[Path],
    *,
    config: RunnerConfig,
    lane: str,
    policy: str,
) -> None:
    bindings = plan.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise RunnerError(f"loop plan calibration bindings are missing: {plan_path}")
    expected_paths = {path.resolve(): path for path in calibration_paths}
    binding_by_variant: dict[str, Mapping[str, object]] = {}
    calibration_maps: dict[str, dict[str, int]] = {}
    calibration_payloads: dict[str, Mapping[str, object]] = {}
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise RunnerError(f"loop plan binding {index} is malformed: {plan_path}")
        variant = binding.get("variant")
        raw_path = binding.get("path")
        relative_path = binding.get("relative_path")
        digest = binding.get("sha256")
        if variant not in {"baseline", "candidate"} or variant in binding_by_variant:
            raise RunnerError(f"loop plan binding variants are invalid: {plan_path}")
        if not isinstance(raw_path, str) or not raw_path:
            raise RunnerError(f"loop plan binding path is missing: {plan_path}")
        resolved = Path(raw_path).expanduser().resolve()
        if resolved not in expected_paths:
            raise RunnerError(
                f"loop plan binding {variant} path is not one of the exact runner calibration paths: {plan_path}"
            )
        expected_relative = Path(
            os.path.relpath(resolved, start=plan_path.resolve().parent)
        ).as_posix()
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or relative_path != expected_relative
        ):
            raise RunnerError(
                f"loop plan binding {variant} relative path mismatch: {plan_path}"
            )
        if not isinstance(digest, str) or digest != _sha256_file(resolved):
            raise RunnerError(
                f"loop plan binding {variant} calibration file hash mismatch: {plan_path}"
            )
        actual = _strict_load(resolved)
        if not isinstance(actual, Mapping):
            raise RunnerError(
                f"calibration binding {variant} is not an object: {resolved}"
            )
        if actual.get("schema") != CALIBRATION_SCHEMA:
            raise RunnerError(
                f"calibration binding {variant} schema mismatch: {resolved}"
            )
        if actual.get("status") != "calibration_only":
            raise RunnerError(
                f"calibration binding {variant} status mismatch: {resolved}"
            )
        _validate_calibration_provenance(resolved, actual)
        if actual.get("variant") != variant:
            raise RunnerError(
                f"calibration binding {variant} variant mismatch: {resolved}"
            )
        if actual.get("backend") != config.backend:
            raise RunnerError(
                f"calibration binding {variant} backend mismatch: {resolved}"
            )
        if actual.get("evidence_lane") != lane or actual.get("input_policy") != policy:
            raise RunnerError(
                f"calibration binding {variant} scope mismatch: {resolved}"
            )
        if actual.get("artifact_kind") != ARTIFACT_KINDS[config.backend]:
            raise RunnerError(
                f"calibration binding {variant} artifact-kind mismatch: {resolved}"
            )
        workload = actual.get("workload")
        expected_workload = {
            "name": config.workload,
            "rows": config.rows,
            "features": config.features,
            "candidates": config.candidates,
            "arity": config.arity,
            "mi_bins": config.mi_bins,
            "top_k": config.top_k,
        }
        if not isinstance(workload, Mapping) or any(
            workload.get(key) != value for key, value in expected_workload.items()
        ):
            raise RunnerError(
                f"calibration binding {variant} workload mismatch: {resolved}"
            )
        for key, expected in _calibration_binding_view(actual).items():
            if binding.get(key) != expected:
                raise RunnerError(
                    f"loop plan binding {variant} metadata mismatch for {key}: {plan_path}"
                )
        binding_by_variant[str(variant)] = binding
        calibration_payloads[str(variant)] = actual
        calibration_maps[str(variant)] = _calibration_entry_map(resolved, actual)
    if set(binding_by_variant) != {"baseline", "candidate"}:
        raise RunnerError(
            f"loop plan must bind baseline and candidate calibrations: {plan_path}"
        )
    expected_scope = plan.get("scope")
    if not isinstance(expected_scope, Mapping):
        raise RunnerError(f"loop plan scope is missing: {plan_path}")
    for variant, actual in calibration_payloads.items():
        if _calibration_scope_view(actual) != dict(expected_scope):
            raise RunnerError(
                f"loop plan calibration {variant} scope does not match plan scope: {plan_path}"
            )
    factor = plan.get("headroom_factor")
    cap = plan.get("max_loop_count")
    if (
        isinstance(factor, bool)
        or not isinstance(factor, int)
        or factor < 1
        or isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap < 1
    ):
        raise RunnerError(f"loop plan headroom policy is invalid: {plan_path}")
    baseline = calibration_maps["baseline"]
    candidate = calibration_maps["candidate"]
    if set(baseline) != set(candidate):
        raise RunnerError(f"loop plan calibration key sets differ: {plan_path}")
    plan_entries = plan.get("entries")
    if not isinstance(plan_entries, list):
        raise RunnerError(f"loop plan entries are missing: {plan_path}")
    plan_map = {
        str(item["key"]): item["loop_count"]
        for item in plan_entries
        if isinstance(item, Mapping) and isinstance(item.get("key"), str)
    }
    if set(plan_map) != set(baseline) or plan.get("entry_count") != len(plan_map):
        raise RunnerError(
            f"loop plan entry keys do not match calibrations: {plan_path}"
        )
    for key in sorted(baseline):
        expected = max(baseline[key], candidate[key]) * factor
        if expected > cap or plan_map.get(key) != expected:
            raise RunnerError(
                f"loop plan entry {key!r} is not max calibration count times fixed headroom: {plan_path}"
            )


def _validate_plan(
    path: Path,
    *,
    config: RunnerConfig,
    lane: str,
    policy: str,
    calibration_paths: Sequence[Path],
) -> dict[str, object]:
    """Validate the immutable plan without rewriting its signed scope.

    ``native_loop_plan.py`` owns plan construction and digest semantics.  The
    runner deliberately treats its output as immutable: replacing its scope
    after construction would discard device and scope_id bindings needed by
    the native helpers and would make the plan a runner-authored, rather than
    calibration-authored, artifact.
    """

    raw = _strict_load(path)
    if not isinstance(raw, Mapping) or raw.get("schema") != LOOP_PLAN_SCHEMA:
        raise RunnerError(f"loop plan schema mismatch: {path}")
    payload = dict(raw)
    try:
        canonical = _canonical_json(payload)
        if path.read_text(encoding="utf-8") != canonical:
            raise RunnerError(f"loop plan file is not canonical JSON: {path}")
    except OSError as exc:
        raise RunnerError(f"cannot read loop plan bytes: {path}: {exc}") from exc
    if payload.get("version") != 1 or payload.get("source_count") != 2:
        raise RunnerError(f"loop plan version/source-count mismatch: {path}")
    if payload.get("variants") != ["baseline", "candidate"]:
        raise RunnerError(f"loop plan variants mismatch: {path}")
    commits = payload.get("source_commits")
    if (
        not isinstance(commits, list)
        or len(commits) != 2
        or any(
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in commit)
            for commit in commits
        )
        or len(set(commits)) != 2
    ):
        raise RunnerError(
            f"loop plan source commits are not distinct/full length: {path}"
        )
    bindings = payload.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise RunnerError(f"loop plan bindings are missing: {path}")
    binding_variants: set[str] = set()
    binding_commits: set[str] = set()
    harness_commits: set[str] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise RunnerError(f"loop plan binding {index} is malformed: {path}")
        variant = binding.get("variant")
        source_commit = binding.get("source_commit")
        product_commit = binding.get("product_source_commit")
        harness_commit = binding.get("harness_source_commit")
        if variant not in {"baseline", "candidate"}:
            raise RunnerError(f"loop plan binding {index} variant is invalid: {path}")
        if (
            not isinstance(source_commit, str)
            or source_commit not in commits
            or product_commit != source_commit
        ):
            raise RunnerError(
                f"loop plan binding {index} product commit is invalid: {path}"
            )
        if (
            not isinstance(harness_commit, str)
            or len(harness_commit) != 40
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in harness_commit
            )
        ):
            raise RunnerError(
                f"loop plan binding {index} harness commit is invalid: {path}"
            )
        binding_variants.add(variant)
        binding_commits.add(source_commit)
        harness_commits.add(harness_commit)
    if (
        binding_variants != {"baseline", "candidate"}
        or binding_commits != set(commits)
        or len(harness_commits) != 1
    ):
        raise RunnerError(f"loop plan binding identities are inconsistent: {path}")
    declared_digest = payload.get("plan_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise RunnerError(f"loop plan digest is missing: {path}")
    if declared_digest != _plan_digest(payload):
        raise RunnerError(f"loop plan digest mismatch: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RunnerError(f"loop plan entries are missing: {path}")
    keys: set[str] = set()
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise RunnerError(f"loop plan entry {index} is malformed: {path}")
        key = item.get("key")
        count = item.get("loop_count")
        if (
            not isinstance(key, str)
            or not key
            or key in keys
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise RunnerError(f"loop plan entry {index} is invalid: {path}")
        keys.add(key)
    if payload.get("entry_count") != len(entries):
        raise RunnerError(f"loop plan entry_count mismatch: {path}")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise RunnerError(f"loop plan scope is missing: {path}")
    if scope.get("backend") != config.backend:
        raise RunnerError(f"loop plan backend scope mismatch: {path}")
    if scope.get("input_policy") != policy:
        raise RunnerError(f"loop plan input-policy scope mismatch: {path}")
    if scope.get("evidence_lane") != lane:
        raise RunnerError(f"loop plan evidence-lane scope mismatch: {path}")
    if scope.get("artifact_kind") != ARTIFACT_KINDS[config.backend]:
        raise RunnerError(f"loop plan artifact-kind scope mismatch: {path}")
    workload = scope.get("workload")
    if not isinstance(workload, Mapping) or workload.get("name") != config.workload:
        raise RunnerError(f"loop plan workload scope mismatch: {path}")
    expected_workload = {
        "rows": config.rows,
        "features": config.features,
        "candidates": config.candidates,
        "arity": config.arity,
        "mi_bins": config.mi_bins,
        "top_k": config.top_k,
    }
    for key, value in expected_workload.items():
        if workload.get(key) != value:
            raise RunnerError(f"loop plan workload field {key!r} mismatch: {path}")
    if not isinstance(scope.get("device"), Mapping) or not scope.get("device"):
        raise RunnerError(f"loop plan device scope is missing: {path}")
    if not isinstance(scope.get("scope_id"), str) or not scope.get("scope_id"):
        raise RunnerError(f"loop plan scope_id is missing: {path}")
    _reauthenticate_plan_calibrations(
        path,
        payload,
        calibration_paths,
        config=config,
        lane=lane,
        policy=policy,
    )
    return payload


def _plan(
    config: RunnerConfig,
    *,
    lane: str,
    policy: str,
    calibration_paths: Sequence[Path],
    output: Path,
    dry_run: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(config.loop_plan_script),
        "--calibration",
        str(calibration_paths[0]),
        "--calibration",
        str(calibration_paths[1]),
        "--output",
        str(output),
    ]
    if dry_run:
        return {
            "path": str(output),
            "semantic_sha256": None,
            "file_sha256": None,
            "scope": {
                "backend": config.backend,
                "artifact_kind": ARTIFACT_KINDS[config.backend],
                "evidence_lane": lane,
                "input_policy": policy,
                "workload": config.workload,
            },
            "command": command,
            "dry_run": True,
        }
    _run(command, cwd=config.harness_source_root, environment=os.environ, dry_run=False)
    payload = _validate_plan(
        output,
        config=config,
        lane=lane,
        policy=policy,
        calibration_paths=calibration_paths,
    )
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise RunnerError(f"loop plan scope is missing: {output}")
    return {
        "path": str(output),
        "semantic_sha256": str(payload["plan_sha256"]),
        "file_sha256": _sha256_file(output),
        "scope": dict(scope),
        "command": command,
        "dry_run": False,
    }


def _validate_recorded_output(
    path: Path,
    *,
    config: RunnerConfig,
    variant: str,
    variant_spec: Variant,
    expected_source_commit: str,
    expected_harness_commit: str,
    expected_command: Sequence[str],
    lane: str,
    policy: str,
    block: int,
    sequence: Sequence[str],
    plan: Mapping[str, object],
    expected_environment: Mapping[str, str],
    expected_affinity: Sequence[int],
    expected_child_pid: int,
    dry_run: bool,
) -> dict[str, object]:
    if dry_run:
        return {
            "schema": "dry-run",
            "status": "planned",
            "backend": config.backend,
            "evidence_lane": lane,
            "input_policy": policy,
            "variant": variant,
            "ab_block": block,
            "variant_sequence": list(sequence),
            "process_isolation": PROCESS_ISOLATION,
            "lane_isolation": LANE_ISOLATION,
            "loop_plan": dict(plan),
            "records": [],
        }
    if not path.is_file():
        raise RunnerError(f"recorded helper did not create JSON output: {path}")
    payload = _strict_load(path)
    if not isinstance(payload, Mapping):
        raise RunnerError(f"recorded helper output must be an object: {path}")
    expected = {
        "schema": ARTIFACT_SCHEMAS[config.backend],
        "status": "pass",
        "backend": config.backend,
        "artifact_kind": ARTIFACT_KINDS[config.backend],
        "evidence_lane": lane,
        "input_policy": policy,
        "variant": variant,
        "ab_block": block,
        "variant_sequence": list(sequence),
        "process_isolation": PROCESS_ISOLATION,
        "lane_isolation": LANE_ISOLATION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RunnerError(
                f"recorded output {path} has {key}={payload.get(key)!r}, expected {value!r}"
            )
    expected_load_status = (
        "loaded_canonical_lane_only"
        if lane == "canonical_payload_api"
        else "not_loaded_by_lane_contract"
    )
    if payload.get("payload_load_status") != expected_load_status:
        raise RunnerError(f"recorded output payload-load status mismatch: {path}")
    _validate_lane_contract(payload, path, lane=lane, artifact="recorded output")
    self_checks = payload.get("self_checks")
    expected_absence_attested = lane != "canonical_payload_api"
    if (
        not isinstance(self_checks, Mapping)
        or self_checks.get("payload_absence_attested") is not expected_absence_attested
    ):
        raise RunnerError(
            f"recorded output payload-absence attestation mismatch: {path}"
        )
    profiles = payload.get("profiles")
    if profiles != list(PROFILES):
        raise RunnerError(f"recorded output {path} does not cover exactly {PROFILES}")
    workload = payload.get("workload")
    if not isinstance(workload, Mapping):
        raise RunnerError(f"recorded output workload is missing: {path}")
    expected_workload = {
        "name": config.workload,
        "rows": config.rows,
        "features": config.features,
        "candidates": config.candidates,
        "arity": config.arity,
        "mi_bins": config.mi_bins,
        "top_k": config.top_k,
    }
    for key, value in expected_workload.items():
        if workload.get(key) != value:
            raise RunnerError(
                f"recorded output workload field {key!r} mismatch: {path}"
            )
    device = payload.get("device")
    if not isinstance(device, Mapping) or not device:
        raise RunnerError(f"recorded output device identity is missing: {path}")
    device_ordinal = device.get("ordinal", device.get("id"))
    if device_ordinal is not None and device_ordinal != config.device:
        raise RunnerError(f"recorded output device ordinal mismatch: {path}")
    if device_ordinal is None and not any(
        device.get(key) not in (None, "")
        for key in ("name", "gcn_arch", "compute_major")
    ):
        raise RunnerError(f"recorded output device identity is insufficient: {path}")
    plan_scope = plan.get("scope")
    plan_device = plan_scope.get("device") if isinstance(plan_scope, Mapping) else None
    if not isinstance(plan_device, Mapping) or not plan_device:
        raise RunnerError(f"recorded output loop-plan device scope is missing: {path}")
    # Calibration artifacts intentionally contain only the stable physical
    # identity needed to share one loop plan across baseline/candidate builds.
    # Recorded artifacts may add runtime/driver fields, but every calibrated
    # identity field must still match.  ROCm historically names its selected
    # ordinal `id` in the recorded device object and `ordinal` in calibration.
    for key, value in plan_device.items():
        recorded_key = "id" if key == "ordinal" and "id" in device else key
        if device.get(recorded_key) != value:
            raise RunnerError(
                f"recorded output device differs from its loop-plan scope: {path}"
            )
    if payload.get("scope_id") != plan_scope.get("scope_id"):
        raise RunnerError(
            f"recorded output scope_id differs from its loop plan: {path}"
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RunnerError(f"recorded output source commit is missing: {path}")
    if source_commit != expected_source_commit:
        raise RunnerError(f"recorded output source commit mismatch: {path}")
    if payload.get("product_source_commit") != expected_source_commit:
        raise RunnerError(f"recorded output product source commit mismatch: {path}")
    if payload.get("harness_source_commit") != expected_harness_commit:
        raise RunnerError(f"recorded output harness source commit mismatch: {path}")
    command_line = payload.get("command_line")
    if not isinstance(command_line, list) or list(command_line) != list(
        expected_command
    ):
        raise RunnerError(f"recorded output command line mismatch: {path}")
    for field, expected_path in (
        ("product_source_root", variant_spec.source_root),
        ("source_root", variant_spec.source_root),
        ("harness_source_root", config.harness_source_root),
    ):
        observed_path = payload.get(field)
        if (
            observed_path is not None
            and Path(str(observed_path)).resolve() != expected_path.resolve()
        ):
            raise RunnerError(f"recorded output {field} mismatch: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RunnerError(f"recorded output provenance is missing: {path}")
    for field, expected_path in (
        ("payload", variant_spec.payload),
        ("wheel", variant_spec.wheel),
    ):
        identity = provenance.get(field)
        if not isinstance(identity, Mapping):
            raise RunnerError(f"recorded output provenance.{field} is missing: {path}")
        if Path(str(identity.get("path"))).resolve() != expected_path.resolve():
            raise RunnerError(
                f"recorded output provenance.{field} path mismatch: {path}"
            )
        if identity.get("sha256") != _sha256_file(expected_path):
            raise RunnerError(
                f"recorded output provenance.{field} hash mismatch: {path}"
            )
    binary = provenance.get("benchmark_binary", provenance.get("helper"))
    if not isinstance(binary, Mapping):
        raise RunnerError(
            f"recorded output benchmark binary identity is missing: {path}"
        )
    if (
        Path(str(binary.get("path"))).resolve()
        != Path(str(expected_command[0])).resolve()
    ):
        raise RunnerError(f"recorded output benchmark binary path mismatch: {path}")
    if binary.get("sha256") != _sha256_file(Path(str(expected_command[0]))):
        raise RunnerError(f"recorded output benchmark binary hash mismatch: {path}")
    _validate_process_attestation(
        payload,
        path,
        expected_environment=expected_environment,
        expected_affinity=expected_affinity,
        expected_child_pid=expected_child_pid,
    )
    loop_plan = payload.get("loop_plan")
    if not isinstance(loop_plan, Mapping):
        raise RunnerError(f"recorded output immutable loop plan is missing: {path}")
    if loop_plan.get("mode") != "immutable":
        raise RunnerError(f"recorded output did not use an immutable loop plan: {path}")
    expected_plan_path = str(plan.get("path"))
    expected_plan_relative = Path(
        os.path.relpath(Path(expected_plan_path).resolve(), start=path.resolve().parent)
    ).as_posix()
    if (
        loop_plan.get("path") != expected_plan_path
        or loop_plan.get("relative_path") != expected_plan_relative
        or loop_plan.get("semantic_sha256") != plan.get("semantic_sha256")
        or loop_plan.get("file_sha256") != plan.get("file_sha256")
    ):
        raise RunnerError(f"recorded output loop plan identity mismatch: {path}")
    calibration_prepass = payload.get("calibration_prepass")
    if (
        not isinstance(calibration_prepass, Mapping)
        or calibration_prepass.get("performed") is not True
    ):
        raise RunnerError(f"recorded output calibration coverage is missing: {path}")
    if calibration_prepass.get("uses_shared_calibration_cache") is not True:
        raise RunnerError(
            f"recorded output did not use shared calibration cache: {path}"
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RunnerError(f"recorded helper output has no records: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("evidence_lane") != lane:
            raise RunnerError(f"record {index} in {path} crosses evidence lanes")
    return dict(payload)


def _orchestration_binding(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    helper: Path,
    variant: Variant,
    source_identity: Mapping[str, object],
    harness_identity: Mapping[str, object],
    process_affinity: list[int] | None,
    lane: str,
    policy: str,
    block: int,
    sequence: Sequence[str],
    plan: Mapping[str, object],
    calibration: bool,
    dry_run: bool,
    invocation_id: str,
) -> dict[str, object]:
    snapshot = _environment_snapshot(environment)
    return {
        "command_line": list(command),
        "command_line_sha256": _sha256_bytes(
            _canonical_json(list(command)).encode("utf-8")
        ),
        "cwd": str(cwd),
        "environment": snapshot,
        "environment_sha256": _sha256_bytes(_canonical_json(snapshot).encode("utf-8")),
        "process_affinity": process_affinity,
        "runner_pid": os.getpid(),
        "runner_invocation_id": invocation_id,
        "helper": _file_identity(helper, "helper", dry_run=dry_run),
        "payload": _file_identity(variant.payload, "payload", dry_run=dry_run),
        "wheel": _file_identity(variant.wheel, "wheel", dry_run=dry_run),
        "canonical_evidence": _file_identity(
            variant.canonical_evidence, "canonical evidence", dry_run=dry_run
        ),
        "source": dict(source_identity),
        "harness_source": dict(harness_identity),
        "evidence_lane": lane,
        "input_policy": policy,
        "ab_block": block,
        "variant_sequence": list(sequence),
        "plan": dict(plan),
        "calibration_only": calibration,
        "process_isolation": (
            CALIBRATION_PROCESS_ISOLATION if calibration else PROCESS_ISOLATION
        ),
        "lane_isolation": LANE_ISOLATION,
    }


def _variant_manifest(
    config: RunnerConfig,
    variant: Variant,
    source_identity: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
    plans: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    source_commit = source_identity.get("commit")
    if not isinstance(source_commit, str):
        source_commit = "dry-run-unverified"
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "validated",
        # This field means the runner completed the collection and basic
        # identity gates.  perf13 remains the authority for arithmetic/order
        # claim readiness and will revalidate every artifact.
        "arithmetic_claims_valid": True,
        "claim_scope": "raw lane-isolated native evidence; perf13 owns final claim validation",
        "backend": config.backend,
        "source_commit": source_commit,
        "required_evidence_lanes": list(LANES),
        "required_input_policies": list(config.input_policies),
        "process_isolation": PROCESS_ISOLATION,
        "lane_isolation": LANE_ISOLATION,
        "calibration_process_isolation": CALIBRATION_PROCESS_ISOLATION,
        "plans": {key: dict(value) for key, value in plans.items()},
        "artifacts": [dict(item) for item in artifacts],
    }


def run(config: RunnerConfig, *, dry_run: bool = False) -> dict[str, object]:
    _validate_config(config)
    if config.loop_plan_script.is_file() is False and not dry_run:
        raise RunnerError(f"native_loop_plan.py is missing: {config.loop_plan_script}")

    variants = (config.baseline, config.candidate)
    harness_identity = _source_identity(
        config.harness_source_root, "common harness source root", dry_run=dry_run
    )
    source_identities = {
        variant.name: _source_identity(
            variant.source_root, f"{variant.name} source root", dry_run=dry_run
        )
        for variant in variants
    }
    # Validate all immutable external inputs before starting any helper.  This
    # prevents a half-collected matrix when a later lane has a missing file.
    for variant in variants:
        variant.helper_for(
            config.backend, "supplemental_internal_kernel", dry_run=dry_run
        )
        for lane in LANES:
            helper = variant.helper_for(config.backend, lane, dry_run=dry_run)
            _file_identity(helper, f"{variant.name} {lane} helper", dry_run=dry_run)
        _file_identity(variant.payload, f"{variant.name} payload", dry_run=dry_run)
        _file_identity(variant.wheel, f"{variant.name} wheel", dry_run=dry_run)
        _file_identity(
            variant.canonical_evidence,
            f"{variant.name} canonical lifecycle evidence",
            dry_run=dry_run,
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs: list[Path] = []
    for policy in config.input_policies:
        for lane in LANES:
            for variant in variants:
                calibration = (
                    config.output_dir
                    / "calibration"
                    / f"{config.backend}-{lane}-{policy}-{variant.name}.json"
                )
                expected_outputs.append(calibration)
                if config.backend == "cuda":
                    expected_outputs.append(calibration.with_suffix(".csv"))
            expected_outputs.append(
                config.output_dir
                / "calibration"
                / f"{config.backend}-{lane}-{policy}-plan.json"
            )
            for block in (0, 1):
                for variant_name in ("baseline", "candidate"):
                    artifact = (
                        config.output_dir
                        / "artifacts"
                        / f"{config.backend}-{lane}-{policy}-block-{block}-{variant_name}.json"
                    )
                    expected_outputs.append(artifact)
                    if config.backend == "cuda":
                        expected_outputs.append(artifact.with_suffix(".csv"))
    expected_outputs.extend(
        config.output_dir / f"{config.backend}-native-evidence-{name}.json"
        for name in ("baseline", "candidate")
    )
    expected_outputs.append(config.output_dir / f"{config.backend}-native-ab-run.json")
    _reject_stale_outputs(expected_outputs, dry_run=dry_run)
    # The native helpers open their output paths directly.  Create the
    # runner-owned layout before launching the first calibration process so a
    # fresh evidence root behaves the same as an existing one.
    (config.output_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    expected_affinity = (
        list(config.affinity) if config.affinity is not None else _affinity()
    )
    if expected_affinity is None and not dry_run:
        raise RunnerError(
            "native GPU evidence requires an observable process affinity mask"
        )
    # All helper children inherit a Git-scrubbed environment.  This prevents a
    # caller's GIT_DIR/GIT_WORK_TREE/config/object redirection from changing
    # either helper provenance or any subprocess Git probe.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }

    def variant_environment(
        variant: Variant, invocation_id: str | None = None
    ) -> dict[str, str]:
        # Start every child from the same immutable parent snapshot and bind
        # only that child's wheel/payload.  Do not mutate a shared mapping as
        # the matrix advances from baseline to candidate.
        child_environment = dict(environment)
        child_environment["GAFIME_WHEEL_PATH"] = str(variant.wheel)
        child_environment[
            "GAFIME_CUDA_V1_LIB" if config.backend == "cuda" else "GAFIME_ROCM_V1_LIB"
        ] = str(variant.payload)
        if invocation_id is not None:
            child_environment["GAFIME_NATIVE_RUNNER_INVOCATION_ID"] = invocation_id
            child_environment["GAFIME_NATIVE_RUNNER_PID"] = str(os.getpid())
        return child_environment

    calibration_records: list[dict[str, object]] = []
    calibration_paths: dict[tuple[str, str, str], Path] = {}
    plans: dict[tuple[str, str], dict[str, object]] = {}
    manifests_artifacts: dict[str, list[dict[str, object]]] = {
        "baseline": [],
        "candidate": [],
    }

    for policy in config.input_policies:
        for lane in LANES:
            for variant in variants:
                helper = variant.helper_for(config.backend, lane, dry_run=dry_run)
                calibration_path = (
                    config.output_dir
                    / "calibration"
                    / f"{config.backend}-{lane}-{policy}-{variant.name}.json"
                )
                invocation_id = secrets.token_hex(16)
                command = _helper_args(
                    config,
                    variant,
                    helper,
                    lane=lane,
                    input_policy=policy,
                    output=calibration_path,
                    calibration_only=True,
                    variant_name=variant.name,
                )
                calibration_paths[(policy, lane, variant.name)] = calibration_path
                if not dry_run:
                    calibration_environment = variant_environment(
                        variant, invocation_id
                    )
                    run_result = _run(
                        command,
                        cwd=config.harness_source_root,
                        environment=calibration_environment,
                        process_affinity=expected_affinity,
                        dry_run=False,
                    )
                    _validate_calibration(
                        calibration_path,
                        config=config,
                        variant_spec=variant,
                        expected_source_commit=str(
                            source_identities[variant.name].get("commit", "")
                        ),
                        expected_harness_commit=str(harness_identity.get("commit", "")),
                        expected_command=command,
                        expected_environment=calibration_environment,
                        expected_affinity=expected_affinity,
                        expected_child_pid=int(run_result["child_pid"]),
                        lane=lane,
                        policy=policy,
                        variant=variant.name,
                    )
                calibration_identity = _file_identity(
                    calibration_path,
                    "calibration artifact",
                    dry_run=dry_run,
                )
                calibration_binding = _orchestration_binding(
                    command,
                    cwd=config.harness_source_root,
                    environment=variant_environment(variant, invocation_id),
                    helper=helper,
                    variant=variant,
                    source_identity=source_identities[variant.name],
                    harness_identity=harness_identity,
                    process_affinity=expected_affinity,
                    lane=lane,
                    policy=policy,
                    block=-1,
                    sequence=(),
                    plan={
                        "calibration_path": str(calibration_path),
                        "calibration_sha256": calibration_identity["sha256"],
                    },
                    calibration=True,
                    dry_run=dry_run,
                    invocation_id=invocation_id,
                )
                calibration_records.append(
                    {
                        "variant": variant.name,
                        "backend": config.backend,
                        "evidence_lane": lane,
                        "input_policy": policy,
                        "path": str(calibration_path),
                        "sha256": calibration_identity["sha256"],
                        "command_line": command,
                        "process_isolation": CALIBRATION_PROCESS_ISOLATION,
                        "lane_isolation": LANE_ISOLATION,
                        "runner_invocation_id": invocation_id,
                        "orchestration": calibration_binding,
                    }
                )

            plan_path = (
                config.output_dir
                / "calibration"
                / f"{config.backend}-{lane}-{policy}-plan.json"
            )
            plan_info = _plan(
                config,
                lane=lane,
                policy=policy,
                calibration_paths=(
                    calibration_paths[(policy, lane, "baseline")],
                    calibration_paths[(policy, lane, "candidate")],
                ),
                output=plan_path,
                dry_run=dry_run,
            )
            plans[(policy, lane)] = plan_info

    recorded_commands: list[list[str]] = []
    artifacts: list[dict[str, object]] = []
    for policy in config.input_policies:
        for lane in LANES:
            plan_info = plans[(policy, lane)]
            plan_path = Path(str(plan_info["path"]))
            for block in (0, 1):
                sequence = _variant_sequence(block)
                for variant_name in sequence:
                    variant = (
                        config.baseline
                        if variant_name == "baseline"
                        else config.candidate
                    )
                    helper = variant.helper_for(config.backend, lane, dry_run=dry_run)
                    invocation_id = secrets.token_hex(16)
                    output = (
                        config.output_dir
                        / "artifacts"
                        / f"{config.backend}-{lane}-{policy}-block-{block}-{variant_name}.json"
                    )
                    command = _helper_args(
                        config,
                        variant,
                        helper,
                        lane=lane,
                        input_policy=policy,
                        output=output,
                        calibration_only=False,
                        plan=plan_path,
                        variant_name=variant_name,
                        block=block,
                        sequence=sequence,
                    )
                    recorded_commands.append(command)
                    child_pid = -1
                    result_environment = variant_environment(variant, invocation_id)
                    if not dry_run:
                        run_result = _run(
                            command,
                            cwd=config.harness_source_root,
                            environment=result_environment,
                            process_affinity=expected_affinity,
                            dry_run=False,
                        )
                        child_pid = int(run_result["child_pid"])
                    payload = _validate_recorded_output(
                        output,
                        config=config,
                        variant=variant_name,
                        variant_spec=variant,
                        expected_source_commit=str(
                            source_identities[variant_name].get("commit", "")
                        ),
                        expected_harness_commit=str(harness_identity.get("commit", "")),
                        expected_command=command,
                        lane=lane,
                        policy=policy,
                        block=block,
                        sequence=sequence,
                        plan=plan_info,
                        expected_environment=result_environment,
                        expected_affinity=expected_affinity,
                        expected_child_pid=child_pid,
                        dry_run=dry_run,
                    )
                    identity = _file_identity(
                        output,
                        "recorded native artifact",
                        dry_run=dry_run,
                    )
                    binding = _orchestration_binding(
                        command,
                        cwd=config.harness_source_root,
                        environment=variant_environment(variant, invocation_id),
                        helper=helper,
                        variant=variant,
                        source_identity=source_identities[variant_name],
                        harness_identity=harness_identity,
                        process_affinity=expected_affinity,
                        lane=lane,
                        policy=policy,
                        block=block,
                        sequence=sequence,
                        plan=plan_info,
                        calibration=False,
                        dry_run=dry_run,
                        invocation_id=invocation_id,
                    )
                    artifact = {
                        "variant": variant_name,
                        "backend": config.backend,
                        "kind": ARTIFACT_KINDS[config.backend],
                        "input_policy": policy,
                        "path": str(output),
                        "sha256": identity["sha256"],
                        "schedule": {
                            "ab_block": block,
                            "variant": variant_name,
                            "variant_sequence": list(sequence),
                            "evidence_lane": lane,
                            "input_policy": policy,
                            "process_isolation": PROCESS_ISOLATION,
                            "loop_plan": {
                                "path": str(plan_info["path"]),
                                "relative_path": Path(
                                    os.path.relpath(
                                        Path(str(plan_info["path"])).resolve(),
                                        start=output.resolve().parent,
                                    )
                                ).as_posix(),
                                "semantic_sha256": plan_info["semantic_sha256"],
                                "file_sha256": plan_info["file_sha256"],
                            },
                            "runner_invocation_id": invocation_id,
                        },
                        "orchestration": binding,
                        "payload_identity": payload.get("provenance", {}).get("payload")
                        if isinstance(payload.get("provenance"), Mapping)
                        else None,
                    }
                    artifacts.append(artifact)
                    manifests_artifacts[variant_name].append(artifact)

    if len(calibration_records) != EXPECTED_CALIBRATION_PROCESSES:
        raise RunnerError(
            f"expected {EXPECTED_CALIBRATION_PROCESSES} calibration processes, got {len(calibration_records)}"
        )
    if len(recorded_commands) != EXPECTED_RECORDED_PROCESSES:
        raise RunnerError(
            f"expected {EXPECTED_RECORDED_PROCESSES} recorded processes, got {len(recorded_commands)}"
        )
    if not dry_run:
        paths = [str(item["path"]) for item in artifacts]
        hashes = [str(item["sha256"]) for item in artifacts]
        if len(set(paths)) != len(paths):
            raise RunnerError(
                "recorded native artifact path reused across fresh processes"
            )
        if len(set(hashes)) != len(hashes):
            raise RunnerError(
                "recorded native artifact hash reused across fresh processes"
            )

    manifest_paths: dict[str, str] = {}
    for variant in variants:
        manifest_path = (
            config.output_dir / f"{config.backend}-native-evidence-{variant.name}.json"
        )
        manifest = _variant_manifest(
            config,
            variant,
            source_identities[variant.name],
            manifests_artifacts[variant.name],
            {
                f"{policy}/{lane}": plans[(policy, lane)]
                for policy in config.input_policies
                for lane in LANES
            },
        )
        if not dry_run:
            _write_json(manifest_path, manifest)
        manifest_paths[variant.name] = str(manifest_path)

    summary: dict[str, object] = {
        "schema": RUNNER_SCHEMA,
        "status": "dry_run" if dry_run else "pass",
        "backend": config.backend,
        "workload": config.workload,
        "input_policies": list(config.input_policies),
        "evidence_lanes": list(LANES),
        "process_isolation": PROCESS_ISOLATION,
        "calibration_process_isolation": CALIBRATION_PROCESS_ISOLATION,
        "calibration_process_count": len(calibration_records),
        "recorded_process_count": len(recorded_commands),
        "expected_recorded_process_count": EXPECTED_RECORDED_PROCESSES,
        "plans": {
            f"{policy}/{lane}": plans[(policy, lane)]
            for policy in config.input_policies
            for lane in LANES
        },
        "calibrations": calibration_records,
        "recorded_commands": recorded_commands,
        "artifacts": artifacts,
        "manifests": manifest_paths,
        "process_affinity": expected_affinity,
        "harness_source": harness_identity,
        "publication": {"performed": False, "actions": []},
    }
    if not dry_run:
        summary_path = config.output_dir / f"{config.backend}-native-ab-run.json"
        _write_json(summary_path, summary)
        summary["summary_path"] = str(summary_path)
    return summary


def _variant_from_args(
    args: argparse.Namespace,
    name: str,
    *,
    dry_run: bool,
) -> Variant:
    prefix = name.replace("-", "_")
    lane_values = getattr(args, f"{prefix}_helper_lane")
    lane_helpers = _parse_lane_bindings(lane_values)
    return Variant(
        name=name,
        helper=Path(getattr(args, f"{prefix}_helper")).expanduser().resolve(),
        payload=Path(getattr(args, f"{prefix}_payload")).expanduser().resolve(),
        wheel=Path(getattr(args, f"{prefix}_wheel")).expanduser().resolve(),
        source_root=Path(getattr(args, f"{prefix}_source_root")).expanduser().resolve(),
        canonical_evidence=Path(getattr(args, f"{prefix}_canonical_evidence"))
        .expanduser()
        .resolve(),
        lane_helpers=lane_helpers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--input-policy", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--harness-source-root", required=True, type=Path)
    parser.add_argument(
        "--loop-plan-script",
        type=Path,
        default=Path(__file__).with_name("native_loop_plan.py"),
    )
    for name in ("baseline", "candidate"):
        parser.add_argument(f"--{name}-helper", required=True)
        parser.add_argument(f"--{name}-helper-lane", action="append", default=[])
        parser.add_argument(f"--{name}-payload", required=True)
        parser.add_argument(f"--{name}-wheel", required=True)
        parser.add_argument(f"--{name}-source-root", required=True)
        parser.add_argument(f"--{name}-canonical-evidence", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--arity", type=int, default=1)
    parser.add_argument("--mi-bins", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--order-repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--dataset-seed", type=int, default=0x524F434D31312D31)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--affinity",
        required=True,
        help="one CPU number; every fresh helper is pinned to that CPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_policies = _split_policies(args.input_policy)
        config = RunnerConfig(
            backend=args.backend,
            workload=args.workload,
            input_policies=input_policies,
            baseline=_variant_from_args(args, "baseline", dry_run=args.dry_run),
            candidate=_variant_from_args(args, "candidate", dry_run=args.dry_run),
            harness_source_root=args.harness_source_root.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            loop_plan_script=args.loop_plan_script.expanduser().resolve(),
            rows=args.rows,
            features=args.features,
            candidates=args.candidates,
            arity=args.arity,
            mi_bins=args.mi_bins,
            top_k=args.top_k,
            warmups=args.warmups,
            repeats=args.repeats,
            order_repetitions=args.order_repetitions,
            seed=args.seed,
            dataset_seed=args.dataset_seed,
            device=args.device,
            affinity=_parse_affinity(args.affinity),
        )
        summary = run(config, dry_run=args.dry_run)
    except RunnerError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
