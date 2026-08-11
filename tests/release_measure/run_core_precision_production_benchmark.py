#!/usr/bin/env python3
"""Compile and run the tracked production Core precision executor harness.

Unlike the supplemental direct-kernel diagnostic, this runner starts a fresh
child process for every requested product cell.  Each child links one exact
product set of Core/orchestrator/types/Rayon rlibs and measures the real
planner -> resident matrix -> `PrecisionComputeBackend` -> ranked result path.

The runner constructs one persisted, seeded, balanced profile-order schedule
for each A/B block. Both variants replay the exact schedule within a block;
the second block uses a distinct schedule while reversing variant order.
`--variant`, `--ab-block`, and `--variant-sequence` are embedded in every child
artifact so perf13 can reject a missing reversal or schedule disagreement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import permutations, product
import json
import math
import os
from pathlib import Path
import random
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
from typing import Iterable, Sequence


HARNESS_SOURCE = Path("tests/release_measure/core_precision_production_benchmark.rs")
HARNESS_RUNNER = Path("tests/release_measure/run_core_precision_production_benchmark.py")
RESULT_SCHEMA = "gafime.core-production-executor.v1"
NATIVE_EVIDENCE_SCHEMA = "gafime.precision-profile-native-evidence.v1"
PROCESS_ISOLATION = "fresh_helper_process_per_variant_trial"
FROZEN_PRE_REPAIR_BASELINE_SHA = "d52199f44aa80ab8ef50c18db95dd1630961cdaf"
RELEASE_WARMUPS = 10
RELEASE_REPETITIONS = 30
MIN_MEASURED_REGION_NS = 100_000_000
CALIBRATION_TARGET_REGION_NS = 200_000_000
CALIBRATION_PREFLIGHT_SAMPLES = 3
MAX_CALIBRATION_REFINEMENTS = 4
MAX_LOOP_COUNT = 1_048_576
METRIC_IDS = {"pearson": 1, "spearman": 2, "mutual_info": 3, "r2": 4}
SNAPSHOT_PARITY_POLICY = {
    "structural_metadata": "exact",
    "fp32_values": "bit_exact",
    "mutual_info_values": "bit_exact_all_profiles",
    "mixed_other_finite_values": "absolute_only_1e-12",
    "fp64_other_finite_values": "absolute_only_2e-12",
    "non_finite_classification": "exact",
}


@dataclass(frozen=True)
class ScalingExecutionPlan:
    """Truthful bounded Core thread-scaling plan for the current CPU set."""

    allowed_cpu_count: int
    requested_worker_modes: tuple[str, ...]
    executed_worker_modes: tuple[str, ...]
    skipped_worker_modes: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CellScheduleEntry:
    schedule_index: int
    input_policy: str
    workload: str
    metric: str
    profile: str
    worker_mode: str
    profile_order: tuple[str, ...]
    profile_order_ordinal: int


@dataclass(frozen=True)
class RepositoryIdentity:
    root: Path
    commit: str
    tree: str
    git_dir: Path
    git_common_dir: Path


@dataclass(frozen=True)
class GitIdentity:
    executable: Path
    sha256: str
    version: str
    trusted_path: str
    sanitized_environment_variables: tuple[str, ...]


@dataclass(frozen=True)
class SourceIdentity:
    path: Path
    relative_path: Path
    sha256: str
    git_blob: str


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size_bytes: int


_TRUSTED_GIT_DIRECTORIES = (
    Path("/usr/bin"),
    Path("/bin"),
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
    Path("/opt/local/bin"),
)


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> FileIdentity:
    resolved = path.resolve(strict=True)
    return FileIdentity(resolved, _sha256(resolved), resolved.stat().st_size)


def _resolve_git_executable() -> Path:
    for directory in _TRUSTED_GIT_DIRECTORIES:
        candidate = directory / "git"
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    trusted_path = os.pathsep.join(str(path) for path in _TRUSTED_GIT_DIRECTORIES)
    discovered = shutil.which("git", path=trusted_path)
    if discovered:
        resolved = Path(discovered).resolve(strict=True)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise RuntimeError("could not resolve trusted Git from system locations")


def _git_environment(git_executable: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    environment = os.environ.copy()
    inherited = tuple(sorted(key for key in environment if key.startswith("GIT_")))
    for key in inherited:
        environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "PATH": os.pathsep.join(
                [str(git_executable.parent)]
                + [str(path) for path in _TRUSTED_GIT_DIRECTORIES]
            ),
        }
    )
    return environment, inherited


def _benchmark_environment() -> dict[str, str]:
    """Preserve the actual tool/runtime PATH while removing Git control input."""

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key, None)
    return environment


def _git_identity() -> GitIdentity:
    executable = _resolve_git_executable()
    environment, sanitized = _git_environment(executable)
    version = _run([str(executable), "--version"], env=environment).stdout.strip()
    if not version.startswith("git version "):
        raise RuntimeError("trusted Git returned an invalid version")
    return GitIdentity(
        executable=executable,
        sha256=_sha256(executable),
        version=version,
        trusted_path=environment["PATH"],
        sanitized_environment_variables=sanitized,
    )


def _git(root: Path, *arguments: str, git: GitIdentity | None = None) -> str:
    identity = git or _git_identity()
    environment, _ = _git_environment(identity.executable)
    return _run(
        [str(identity.executable), "-C", str(root), *arguments], env=environment
    ).stdout.strip()


def _full_hex(value: str, length: int, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be a {length}-character hexadecimal identity")
    return value.lower()


def _repository_identity(root: Path) -> RepositoryIdentity:
    resolved = root.resolve(strict=True)
    git = _git_identity()
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel", git=git))
    if top_level.resolve(strict=True) != resolved:
        raise ValueError("Git top-level does not match physical repository root")
    git_dir = Path(_git(resolved, "rev-parse", "--git-dir", git=git))
    if not git_dir.is_absolute():
        git_dir = resolved / git_dir
    git_dir = git_dir.resolve(strict=True)
    common_dir = Path(_git(resolved, "rev-parse", "--git-common-dir", git=git))
    if not common_dir.is_absolute():
        common_dir = resolved / common_dir
    common_dir = common_dir.resolve(strict=True)
    dot_git = resolved / ".git"
    if dot_git.is_dir():
        expected_git_dir = dot_git.resolve(strict=True)
        expected_common_dir = expected_git_dir
    elif dot_git.is_file():
        marker = dot_git.read_text(encoding="utf-8").splitlines()
        if not marker or not marker[0].startswith("gitdir:"):
            raise ValueError("invalid linked-worktree .git file")
        linked = Path(marker[0][len("gitdir:") :].strip())
        if not linked.is_absolute():
            linked = dot_git.parent / linked
        expected_git_dir = linked.resolve(strict=True)
        expected_common_dir = (
            expected_git_dir.parent.parent
            if expected_git_dir.parent.name == "worktrees"
            else expected_git_dir.parent
        )
    else:
        raise ValueError("repository has no .git directory or linked-worktree file")
    if git_dir != expected_git_dir or common_dir != expected_common_dir:
        raise ValueError("Git directory provenance does not match physical repository")
    if _git(resolved, "status", "--porcelain=v1", "--untracked-files=all", git=git):
        raise ValueError(f"repository must be clean: {resolved}")
    return RepositoryIdentity(
        root=resolved,
        commit=_full_hex(_git(resolved, "rev-parse", "HEAD", git=git), 40, "commit"),
        tree=_full_hex(_git(resolved, "rev-parse", "HEAD^{tree}", git=git), 40, "tree"),
        git_dir=git_dir,
        git_common_dir=common_dir,
    )


def _tracked_source_identity(root: Path, relative_path: Path) -> SourceIdentity:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("harness path must be repository relative")
    source = (root / relative_path).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("harness source escaped its repository") from error
    _git(root, "ls-files", "--error-unmatch", relative_path.as_posix())
    current_blob = _full_hex(
        _git(root, "hash-object", relative_path.as_posix()), 40, "current source blob"
    )
    head_blob = _full_hex(
        _git(root, "rev-parse", f"HEAD:{relative_path.as_posix()}"),
        40,
        "HEAD source blob",
    )
    if current_blob != head_blob:
        raise ValueError("harness source differs from its checked-in HEAD blob")
    return SourceIdentity(source, relative_path, _sha256(source), head_blob)


def _parse_csv(value: str, *, allowed: set[str], label: str) -> tuple[str, ...]:
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    if not entries:
        raise ValueError(f"{label} must contain at least one value")
    if len(set(entries)) != len(entries):
        raise ValueError(f"{label} must not contain duplicates; order is significant")
    unsupported = [entry for entry in entries if entry not in allowed]
    if unsupported:
        raise ValueError(f"{label} contains unsupported values: {unsupported}")
    return entries


def _parse_worker_modes(value: str) -> tuple[str, ...]:
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    if not entries:
        raise ValueError("--worker-modes must contain at least one mode")
    if len(set(entries)) != len(entries):
        raise ValueError("--worker-modes must not contain duplicates")
    for entry in entries:
        if entry == "default":
            continue
        if not entry.isdecimal() or int(entry) < 1:
            raise ValueError("worker modes must be positive integers or default")
    required = {"1", "default"}
    if not required.issubset(entries):
        raise ValueError("--worker-modes must include 1 and default")
    return entries


def _allowed_cpu_count() -> int:
    """Return the actual CPU-set ceiling rather than host-wide CPU count."""

    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = set()
    if affinity:
        return len(affinity)
    return max(1, os.cpu_count() or 1)


def _scaling_execution_plan(worker_modes: Sequence[str]) -> ScalingExecutionPlan:
    """Skip oversubscribing 2/4-worker diagnostics on a constrained CPU set.

    The requested labels remain in the artifact, while only attainable explicit
    worker counts are executed.  ``default`` always runs and reports its actual
    effective pool size from the child process.
    """

    allowed = _allowed_cpu_count()
    executed: list[str] = []
    skipped: list[dict[str, object]] = []
    for mode in worker_modes:
        if mode == "default" or int(mode) <= allowed:
            executed.append(mode)
        else:
            skipped.append(
                {
                    "worker_mode": mode,
                    "reason": "allowed_cpu_count_below_requested_workers",
                    "allowed_cpu_count": allowed,
                    "requested_workers": int(mode),
                }
            )
    if not executed:
        raise RuntimeError("no Core worker mode can execute on the observed CPU set")
    return ScalingExecutionPlan(
        allowed_cpu_count=allowed,
        requested_worker_modes=tuple(worker_modes),
        executed_worker_modes=tuple(executed),
        skipped_worker_modes=tuple(skipped),
    )


def _cell_schedule(
    *,
    profiles: Sequence[str],
    metrics: Sequence[str],
    workloads: Sequence[str],
    policies: Sequence[str],
    worker_modes: Sequence[str],
    seed: int,
    ab_block: int,
) -> tuple[tuple[CellScheduleEntry, ...], int, str, dict[str, int]]:
    """Build one persisted balanced randomized schedule shared by A and B.

    Every context receives one complete profile order.  The canonical profile
    permutations are assigned with counts differing by at most one, while both
    context order and assignment are deterministically shuffled.  The block is
    mixed into the seed, so A and B agree within a block and block 1 cannot
    silently inherit block 0's fixed order.
    """

    if ab_block < 0:
        raise ValueError("A/B block must be non-negative")
    schedule_seed = (int(seed) ^ ((ab_block + 1) * 0x9E37_79B9_7F4A_7C15)) & (
        (1 << 64) - 1
    )
    randomizer = random.Random(schedule_seed)
    contexts = list(product(policies, workloads, metrics, worker_modes))
    randomizer.shuffle(contexts)
    orders = tuple(permutations(profiles))
    if not orders:
        raise ValueError("production schedule requires at least one profile")
    assigned_orders = [orders[index % len(orders)] for index in range(len(contexts))]
    randomizer.shuffle(assigned_orders)
    entries: list[CellScheduleEntry] = []
    counts = {"/".join(order): 0 for order in orders}
    for context, profile_order in zip(contexts, assigned_orders, strict=True):
        policy, workload, metric, worker_mode = context
        ordinal = orders.index(profile_order)
        counts["/".join(profile_order)] += 1
        for profile in profile_order:
            entries.append(
                CellScheduleEntry(
                    schedule_index=len(entries),
                    input_policy=policy,
                    workload=workload,
                    metric=metric,
                    profile=profile,
                    worker_mode=worker_mode,
                    profile_order=tuple(profile_order),
                    profile_order_ordinal=ordinal,
                )
            )
    unsigned = {
        "algorithm": "seeded_balanced_profile_orders_v1",
        "seed": schedule_seed,
        "ab_block": ab_block,
        "entries": [
            {
                "schedule_index": entry.schedule_index,
                "input_policy": entry.input_policy,
                "workload": entry.workload,
                "metric": entry.metric,
                "profile": entry.profile,
                "worker_mode": entry.worker_mode,
                "profile_order": list(entry.profile_order),
                "profile_order_ordinal": entry.profile_order_ordinal,
            }
            for entry in entries
        ],
    }
    schedule_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(entries), schedule_seed, schedule_hash, counts


def _compiler_command(
    *,
    rustup: str,
    toolchain: str,
    source: Path,
    product_rlib: Path,
    orchestrator_rlib: Path,
    types_rlib: Path,
    rayon_rlib: Path,
    dependency_dirs: Iterable[Path],
    binary: Path,
) -> list[str]:
    command = [
        rustup,
        "run",
        toolchain,
        "rustc",
        "--crate-name",
        "gafime_core_precision_production_benchmark",
        "--edition=2021",
        str(source),
        "--extern",
        f"gafime_cpu={product_rlib}",
        "--extern",
        f"gafime_orchestrator={orchestrator_rlib}",
        "--extern",
        f"gafime_types={types_rlib}",
        "--extern",
        f"rayon={rayon_rlib}",
    ]
    for directory in dependency_dirs:
        command.extend(("-L", f"dependency={directory}"))
    command.extend(
        (
            "-Copt-level=3",
            "-Ccodegen-units=1",
            "-Clto=fat",
            "-Cembed-bitcode=yes",
            "-o",
            str(binary),
        )
    )
    return command


def _compiler_environment(
    source: SourceIdentity,
    runner_source: SourceIdentity,
    product_rlib: FileIdentity,
    orchestrator_rlib: FileIdentity,
    types_rlib: FileIdentity,
    rayon_rlib: FileIdentity,
    compiler_command: Sequence[str],
) -> dict[str, str]:
    environment = _benchmark_environment()
    command_json = json.dumps(list(compiler_command), separators=(",", ":"), ensure_ascii=True)
    environment.update(
        {
            "GAFIME_COMPILED_HARNESS_SOURCE_SHA256": source.sha256,
            "GAFIME_COMPILED_HARNESS_SOURCE_GIT_BLOB": source.git_blob,
            "GAFIME_COMPILED_HARNESS_SOURCE_RELATIVE_PATH": source.relative_path.as_posix(),
            "GAFIME_COMPILED_HARNESS_RUNNER_SHA256": runner_source.sha256,
            "GAFIME_COMPILED_HARNESS_RUNNER_GIT_BLOB": runner_source.git_blob,
            "GAFIME_COMPILED_HARNESS_RUNNER_RELATIVE_PATH": runner_source.relative_path.as_posix(),
            "GAFIME_COMPILED_PRODUCT_RLIB_SHA256": product_rlib.sha256,
            "GAFIME_COMPILED_ORCHESTRATOR_RLIB_SHA256": orchestrator_rlib.sha256,
            "GAFIME_COMPILED_TYPES_RLIB_SHA256": types_rlib.sha256,
            "GAFIME_COMPILED_RAYON_RLIB_SHA256": rayon_rlib.sha256,
            "GAFIME_COMPILED_COMMAND_SHA256": hashlib.sha256(
                command_json.encode("utf-8")
            ).hexdigest(),
        }
    )
    return environment


def _one_line(command: Sequence[str]) -> str:
    completed = _run(command)
    return (completed.stdout or completed.stderr).splitlines()[0].strip()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bootstrap_median_ci(
    values: Sequence[float], *, seed: int, resamples: int
) -> list[float]:
    randomizer = random.Random(seed)
    medians = [
        statistics.median([values[randomizer.randrange(len(values))] for _ in values])
        for _ in range(resamples)
    ]
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _record_statistics(record: dict[str, object], *, seed: int, resamples: int) -> None:
    samples = record.get("samples_ns")
    if not isinstance(samples, list) or not samples:
        raise ValueError("child artifact lacks normalized samples_ns")
    values = [float(value) for value in samples]
    center = statistics.median(values)
    record.update(
        {
            "median_ns_per_call": center,
            "mad_ns_per_call": statistics.median(abs(value - center) for value in values),
            "p05_ns_per_call": _percentile(values, 0.05),
            "p95_ns_per_call": _percentile(values, 0.95),
            "bootstrap_median_95_ci_ns_per_call": _bootstrap_median_ci(
                values, seed=seed, resamples=resamples
            ),
        }
    )


_COMPARABLE_ENVIRONMENT_KEYS = (
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "KMP_HW_SUBSET",
    "MALLOC_ARENA_MAX",
    "RUSTFLAGS",
)


def _canonical_child_environment(child: dict[str, object]) -> dict[str, str]:
    raw = child.get("environment")
    if not isinstance(raw, dict) or not isinstance(raw.get("PATH"), str) or not raw["PATH"]:
        raise RuntimeError("production child must report its actual nonempty PATH")
    if "RAYON_NUM_THREADS" in raw:
        raise RuntimeError("production child inherited forbidden RAYON_NUM_THREADS")
    canonical: dict[str, str] = {}
    for key in _COMPARABLE_ENVIRONMENT_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, str):
            raise RuntimeError(
                "production child environment values must be strings"
            )
        canonical[key] = value
    canonical.update(
        {
            "RAYON_NUM_THREADS": "<scrubbed>",
            "RAYON_NUM_THREADS_POLICY": (
                "removed_by_runner_dedicated_thread_pool_builder_is_authoritative"
            ),
        }
    )
    return canonical


def _canonical_child_device(child: dict[str, object]) -> dict[str, object]:
    raw = child.get("device")
    if not isinstance(raw, dict):
        raise RuntimeError("production child CPU hardware provenance is malformed")
    kind = raw.get("kind")
    identity = raw.get("identity")
    logical = raw.get("logical_cpu_count")
    physical = raw.get("physical_cpu_count")
    if (
        kind != "cpu"
        or not isinstance(identity, str)
        or not identity.strip()
        or not isinstance(logical, int)
        or isinstance(logical, bool)
        or logical < 1
        or (
            physical is not None
            and (
                not isinstance(physical, int)
                or isinstance(physical, bool)
                or physical < 1
                or physical > logical
            )
        )
    ):
        raise RuntimeError("production child CPU hardware provenance is malformed")
    return {
        "kind": kind,
        "identity": identity,
        "logical_cpu_count": logical,
        "physical_cpu_count": physical,
    }


def _canonical_common_child_device(
    children: Sequence[dict[str, object]],
) -> dict[str, object]:
    devices = [_canonical_child_device(child) for child in children]
    if not devices or any(device != devices[0] for device in devices[1:]):
        raise RuntimeError(
            "fresh production children observed inconsistent CPU hardware provenance"
        )
    return devices[0]


def _static_clock_power_view(state: object) -> dict[str, object]:
    """Drop sampled current frequency while retaining comparable power policy."""

    if not isinstance(state, dict):
        raise RuntimeError("production child clock/power state is malformed")
    result: dict[str, object] = {}
    for phase in ("before", "after"):
        raw_phase = state.get(phase)
        if not isinstance(raw_phase, dict):
            raise RuntimeError("production child clock/power phase is malformed")
        raw_policies = raw_phase.get("policy_clock_state")
        if not isinstance(raw_policies, list):
            raise RuntimeError("production child policy clock state is malformed")
        policies = []
        for policy in raw_policies:
            if not isinstance(policy, dict) or not isinstance(policy.get("policy"), str):
                raise RuntimeError("production child CPU policy record is malformed")
            policies.append(
                {
                    key: policy.get(key)
                    for key in (
                        "policy",
                        "scaling_min_freq_khz",
                        "scaling_max_freq_khz",
                        "cpuinfo_min_freq_khz",
                        "cpuinfo_max_freq_khz",
                        "energy_performance_preference",
                    )
                }
            )
        policies.sort(key=lambda item: str(item["policy"]))
        result[phase] = {
            "cpu_governor": raw_phase.get("cpu_governor"),
            "policy_clock_state": policies,
            "platform_power_profile": raw_phase.get("platform_power_profile"),
            "macos_pmset_custom": raw_phase.get("macos_pmset_custom"),
            "power_interface": raw_phase.get("power_interface"),
        }
    return result


def _thread_scaling_tables(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = {}
    for record in records:
        workload = record.get("workload")
        topology = record.get("execution_topology")
        if not isinstance(workload, dict) or not isinstance(topology, dict):
            raise RuntimeError("production scaling record lacks workload/topology")
        key = (
            str(record.get("input_policy")),
            str(workload.get("name")),
            str(record.get("metric")),
            str(record.get("profile")),
        )
        worker_mode = str(topology.get("worker_mode"))
        if worker_mode in groups.setdefault(key, {}):
            raise RuntimeError("duplicate production thread-scaling cell")
        groups[key][worker_mode] = record
    tables: list[dict[str, object]] = []
    for key, cells in sorted(groups.items()):
        one = cells.get("1")
        if one is None:
            raise RuntimeError("thread-scaling table requires its one-worker reference")
        one_median = one.get("median_ns_per_call")
        if not isinstance(one_median, (int, float)) or float(one_median) <= 0.0:
            raise RuntimeError("one-worker scaling median must be positive")
        measurements = []
        for worker_mode, record in sorted(
            cells.items(), key=lambda item: (item[0] == "default", item[0])
        ):
            topology = record["execution_topology"]
            assert isinstance(topology, dict)
            effective = topology.get("effective_rayon_workers")
            median = record.get("median_ns_per_call")
            if (
                not isinstance(effective, int)
                or effective < 1
                or not isinstance(median, (int, float))
                or float(median) <= 0.0
            ):
                raise RuntimeError("thread-scaling measurement is malformed")
            speedup = float(one_median) / float(median)
            measurements.append(
                {
                    "worker_mode": worker_mode,
                    "effective_workers": effective,
                    "median_ns_per_call": float(median),
                    "speedup_vs_one_worker": speedup,
                    "parallel_efficiency": speedup / effective,
                    "measurement_role": topology.get("measurement_role"),
                }
            )
        tables.append(
            {
                "input_policy": key[0],
                "workload": key[1],
                "metric": key[2],
                "profile": key[3],
                "one_worker_median_ns_per_call": float(one_median),
                "measurements": measurements,
                "claim_scope": (
                    "diagnostic thread scaling; default-worker measurement is the "
                    "separate primary production result"
                ),
            }
        )
    return tables


def _identity_json(identity: FileIdentity) -> dict[str, object]:
    return {
        "path": str(identity.path),
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }


def _git_record(git: GitIdentity) -> dict[str, object]:
    return {
        "path": str(git.executable),
        "sha256": git.sha256,
        "version": git.version,
        "trusted_path": git.trusted_path,
        "sanitized_environment_variables": list(git.sanitized_environment_variables),
        "path_lookup_ignored": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-source-root", type=Path, required=True)
    parser.add_argument("--harness-source-root", type=Path, required=True)
    parser.add_argument("--product-rlib", type=Path, required=True)
    parser.add_argument("--orchestrator-rlib", type=Path, required=True)
    parser.add_argument("--types-rlib", type=Path, required=True)
    parser.add_argument("--rayon-rlib", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--profiles", default="fp32,mixed,fp64")
    parser.add_argument("--metrics", default="pearson,spearman,mutual_info,r2")
    parser.add_argument("--workloads", default="latency,medium,kernel")
    parser.add_argument("--input-policies", default="common-f64,native")
    parser.add_argument("--worker-modes", default="1,2,4,default")
    parser.add_argument("--warmups", type=int, default=RELEASE_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=RELEASE_REPETITIONS)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0xC0DE_2026)
    parser.add_argument("--variant", default="candidate")
    parser.add_argument("--ab-block", type=int, default=0)
    parser.add_argument("--variant-sequence", default="candidate")
    parser.add_argument("--mode", choices=("informational", "stable"), default="informational")
    parser.add_argument("--toolchain", default="1.97.1")
    parser.add_argument("--rustup", default="rustup")
    parser.add_argument("--expected-product-commit")
    parser.add_argument("--expected-harness-commit")
    return parser


def _validate_arguments(args: argparse.Namespace) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    if args.warmups < 1 or args.repetitions < 1 or args.bootstrap_resamples < 1:
        raise ValueError("warmups, repetitions, and bootstrap resamples must be positive")
    if args.ab_block < 0:
        raise ValueError("--ab-block must be non-negative")
    profiles = _parse_csv(args.profiles, allowed={"fp32", "mixed", "fp64"}, label="--profiles")
    metrics = _parse_csv(
        args.metrics,
        allowed={"pearson", "spearman", "mutual_info", "r2"},
        label="--metrics",
    )
    workloads = _parse_csv(
        args.workloads, allowed={"latency", "medium", "kernel"}, label="--workloads"
    )
    policies = _parse_csv(
        args.input_policies, allowed={"common-f64", "native"}, label="--input-policies"
    )
    workers = _parse_worker_modes(args.worker_modes)
    sequence = _parse_csv(
        args.variant_sequence,
        allowed={"baseline", "candidate"},
        label="--variant-sequence",
    )
    if args.variant not in sequence:
        raise ValueError("--variant must appear in --variant-sequence")
    if args.mode == "stable":
        if args.warmups < RELEASE_WARMUPS or args.repetitions < RELEASE_REPETITIONS:
            raise ValueError("stable mode requires at least 10 warmups and 30 repetitions")
        if len(sequence) != 2 or set(sequence) != {"baseline", "candidate"}:
            raise ValueError(
                "stable mode requires an explicit baseline,candidate or "
                "candidate,baseline variant sequence; a candidate-only raw artifact "
                "cannot make a comparative performance claim"
            )
        if (
            profiles != ("fp32", "mixed", "fp64")
            or metrics != ("pearson", "spearman", "mutual_info", "r2")
            or workloads != ("latency", "medium", "kernel")
            or policies != ("common-f64", "native")
            or workers != ("1", "2", "4", "default")
        ):
            raise ValueError(
                "stable mode requires the complete canonical 3x4x3x2 matrix "
                "and 1/2/4/default worker modes"
            )
    return profiles, metrics, workloads, policies, workers


def _validate_variant_product_binding(variant: str, product_commit: str) -> None:
    if variant == "baseline":
        if product_commit != FROZEN_PRE_REPAIR_BASELINE_SHA:
            raise ValueError(
                "baseline topology exception requires the frozen pre-repair product commit"
            )
    elif variant == "candidate":
        if product_commit == FROZEN_PRE_REPAIR_BASELINE_SHA:
            raise ValueError(
                "the frozen pre-repair product commit cannot be labeled as the candidate"
            )
    else:
        raise ValueError("--variant must be baseline or candidate")


def _child_environment(
    *,
    base: dict[str, str],
    product: RepositoryIdentity,
    harness: RepositoryIdentity,
    source: SourceIdentity,
    runner_source: SourceIdentity,
    product_rlib: FileIdentity,
    orchestrator_rlib: FileIdentity,
    types_rlib: FileIdentity,
    rayon_rlib: FileIdentity,
    wheel: FileIdentity,
    binary: FileIdentity,
    output: Path,
    profile: str,
    metric: str,
    workload: str,
    policy: str,
    workers: str,
    warmups: int,
    repetitions: int,
    variant: str,
    ab_block: int,
    variant_sequence: Sequence[str],
    runner_invocation_id: str,
    runner_pid: int,
    mode: str,
    schedule_index: int,
    schedule_seed: int,
    schedule_sha256: str,
    profile_order: Sequence[str],
) -> dict[str, str]:
    environment = base.copy()
    # The benchmark owns its dedicated ThreadPoolBuilder configuration.  An
    # inherited global Rayon override would make the requested/default labels
    # untruthful, especially in a constrained CI cpuset.  Preserve its absence
    # in the child provenance rather than silently accepting host policy.
    environment.pop("RAYON_NUM_THREADS", None)
    environment.update(
        {
            "GAFIME_PRODUCTION_PRODUCT_SOURCE_ROOT": str(product.root),
            "GAFIME_PRODUCTION_HARNESS_SOURCE_ROOT": str(harness.root),
            "GAFIME_PRODUCTION_PRODUCT_COMMIT": product.commit,
            "GAFIME_PRODUCTION_PRODUCT_TREE": product.tree,
            "GAFIME_PRODUCTION_HARNESS_COMMIT": harness.commit,
            "GAFIME_PRODUCTION_HARNESS_TREE": harness.tree,
            "GAFIME_PRODUCTION_HARNESS_SOURCE": str(source.path),
            "GAFIME_PRODUCTION_HARNESS_SOURCE_SHA256": source.sha256,
            "GAFIME_PRODUCTION_HARNESS_SOURCE_GIT_BLOB": source.git_blob,
            "GAFIME_PRODUCTION_HARNESS_RUNNER": str(runner_source.path),
            "GAFIME_PRODUCTION_HARNESS_RUNNER_SHA256": runner_source.sha256,
            "GAFIME_PRODUCTION_HARNESS_RUNNER_GIT_BLOB": runner_source.git_blob,
            "GAFIME_PRODUCTION_PRODUCT_RLIB": str(product_rlib.path),
            "GAFIME_PRODUCTION_ORCHESTRATOR_RLIB": str(orchestrator_rlib.path),
            "GAFIME_PRODUCTION_TYPES_RLIB": str(types_rlib.path),
            "GAFIME_PRODUCTION_RAYON_RLIB": str(rayon_rlib.path),
            "GAFIME_PRODUCTION_BENCH_WHEEL": str(wheel.path),
            "GAFIME_PRODUCTION_BENCH_BINARY": str(binary.path),
            "GAFIME_PRODUCTION_BENCH_OUTPUT": str(output),
            "GAFIME_PRODUCTION_PROFILE": profile,
            "GAFIME_PRODUCTION_METRIC": metric,
            "GAFIME_PRODUCTION_WORKLOAD": workload,
            "GAFIME_PRODUCTION_INPUT_POLICY": policy,
            "GAFIME_PRODUCTION_RAYON_WORKERS": workers,
            "GAFIME_PRODUCTION_WARMUPS": str(warmups),
            "GAFIME_PRODUCTION_REPETITIONS": str(repetitions),
            "GAFIME_PRODUCTION_VARIANT": variant,
            "GAFIME_PRODUCTION_AB_BLOCK": str(ab_block),
            "GAFIME_PRODUCTION_VARIANT_SEQUENCE": ",".join(variant_sequence),
            "GAFIME_PRODUCTION_RUNNER_INVOCATION_ID": runner_invocation_id,
            "GAFIME_PRODUCTION_RUNNER_PID": str(runner_pid),
            "GAFIME_PRODUCTION_MODE": mode,
            "GAFIME_PRODUCTION_SCHEDULE_INDEX": str(schedule_index),
            "GAFIME_PRODUCTION_SCHEDULE_SEED": str(schedule_seed),
            "GAFIME_PRODUCTION_SCHEDULE_SHA256": schedule_sha256,
            "GAFIME_PRODUCTION_PROFILE_ORDER": ",".join(profile_order),
        }
    )
    return environment


def _load_child(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"production child did not write valid JSON: {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != "gafime.core-production-executor.child.v1":
        raise RuntimeError("production child schema mismatch")
    return raw


def _rotate_left_u64(value: int, shift: int) -> int:
    mask = 0xFFFF_FFFF_FFFF_FFFF
    value &= mask
    return ((value << shift) | (value >> (64 - shift))) & mask


def _snapshot_digests(snapshot: dict[str, object]) -> tuple[int, int]:
    """Recompute the Rust timed digests from a complete untimed snapshot."""

    row_count = snapshot["row_count"]
    max_arity = snapshot["max_arity"]
    metric_count = snapshot["metric_count"]
    result_dtype = snapshot["result_dtype"]
    result_flags = snapshot["result_flags"]
    metric_ids = snapshot["metric_ids"]
    combo_indices = snapshot["combo_indices"]
    ranks = snapshot["ranks"]
    families = snapshot["families"]
    candidate_ids = snapshot["candidate_ids"]
    row_flags = snapshot["row_flags"]
    metric_value_bits = snapshot["metric_value_bits"]
    assert (
        isinstance(row_count, int)
        and isinstance(max_arity, int)
        and isinstance(metric_count, int)
        and isinstance(result_flags, int)
        and result_dtype in {"f32", "f64"}
        and isinstance(metric_ids, list)
    )
    assert all(
        isinstance(values, list)
        for values in (
            combo_indices,
            ranks,
            families,
            candidate_ids,
            row_flags,
            metric_value_bits,
        )
    )
    structural = (
        _rotate_left_u64(row_count, 17)
        ^ _rotate_left_u64(max_arity, 29)
        ^ _rotate_left_u64(metric_count, 41)
        ^ _rotate_left_u64(result_flags, 53)
        ^ (32 if result_dtype == "f32" else 64)
    )
    for metric_id in metric_ids:
        structural = _rotate_left_u64(structural, 5) ^ int(metric_id)
    for row in range(row_count):
        for feature in combo_indices[row * max_arity : (row + 1) * max_arity]:
            structural = _rotate_left_u64(structural, 5) ^ int(feature)
        structural = _rotate_left_u64(structural, 3) ^ int(ranks[row])
        structural = _rotate_left_u64(structural, 3) ^ int(families[row])
        structural = _rotate_left_u64(structural, 7) ^ int(candidate_ids[row])
        structural = _rotate_left_u64(structural, 3) ^ int(row_flags[row])
    metric = 0
    for bits in metric_value_bits:
        metric = _rotate_left_u64(metric, 11) ^ int(bits)
    return structural, metric


def _snapshot_values_are_self_consistent(snapshot: dict[str, object]) -> bool:
    """Prove text/classes are exact derivations of the timed value bits."""

    result_dtype = snapshot.get("result_dtype")
    value_bits = snapshot.get("metric_value_bits")
    value_text = snapshot.get("metric_value_text")
    value_classes = snapshot.get("metric_value_classes")
    if (
        result_dtype not in {"f32", "f64"}
        or not isinstance(value_bits, list)
        or not isinstance(value_text, list)
        or not isinstance(value_classes, list)
        or len(value_bits) != len(value_text)
        or len(value_bits) != len(value_classes)
    ):
        return False
    bit_width = 32 if result_dtype == "f32" else 64
    integer_format = "!I" if bit_width == 32 else "!Q"
    float_format = "!f" if bit_width == 32 else "!d"
    maximum = (1 << bit_width) - 1
    for bits, text_value, declared_class in zip(
        value_bits, value_text, value_classes, strict=True
    ):
        if (
            not isinstance(bits, int)
            or isinstance(bits, bool)
            or not 0 <= bits <= maximum
            or not isinstance(text_value, str)
            or not isinstance(declared_class, str)
        ):
            return False
        value = struct.unpack(float_format, struct.pack(integer_format, bits))[0]
        actual_class = (
            "nan"
            if math.isnan(value)
            else "positive_infinity"
            if value == math.inf
            else "negative_infinity"
            if value == -math.inf
            else "finite"
        )
        if declared_class != actual_class:
            return False
        try:
            parsed = float(text_value)
        except ValueError:
            return False
        if actual_class == "nan":
            if not math.isnan(parsed):
                return False
        elif actual_class == "positive_infinity":
            if parsed != math.inf:
                return False
        elif actual_class == "negative_infinity":
            if parsed != -math.inf:
                return False
        else:
            try:
                parsed_bits = struct.unpack(
                    integer_format, struct.pack(float_format, parsed)
                )[0]
            except (OverflowError, struct.error):
                return False
            if parsed_bits != bits:
                return False
    return True


def _validate_child_contract(
    child: dict[str, object],
    *,
    expected_variant: str,
    worker_mode: str,
    variant_sequence: Sequence[str],
    runner_pid: int,
    mode: str,
    schedule_entry: CellScheduleEntry,
    schedule_seed: int,
    schedule_sha256: str,
    repetitions: int,
    runner_allowed_cpu_count: int,
) -> None:
    """Reject a child whose declared production topology is not executable truth.

    This remains separate from raw-artifact integrity: a JSON payload can hash
    correctly while still being an invalid default/scaling measurement.
    """

    topology = child.get("execution_topology")
    if not isinstance(topology, dict):
        raise RuntimeError("production child lacks execution topology")
    if expected_variant not in {"baseline", "candidate"}:
        raise RuntimeError("production runner expected variant is unsupported")
    expected_candidate_parallelism = (
        "rayon_candidate_level"
        if expected_variant == "candidate"
        else "frozen_pre_repair_serial_candidate_loop"
    )
    if topology.get("candidate_parallelism") != expected_candidate_parallelism:
        raise RuntimeError(
            "production child candidate topology does not match its product variant"
        )
    expected_semantic_guard = (
        "cfg_test_precision_executor_parallelism_contract"
        if expected_variant == "candidate"
        else "not_applicable_frozen_pre_repair_serial_baseline"
    )
    if topology.get("semantic_candidate_participation_guard") != expected_semantic_guard:
        raise RuntimeError(
            "production child semantic participation guard does not match its product variant"
        )
    expected_role = (
        "primary_default_worker_production_result"
        if worker_mode == "default"
        else "thread_scaling_diagnostic"
    )
    if topology.get("measurement_role") != expected_role:
        raise RuntimeError("production child scaling role does not match requested worker mode")
    if topology.get("worker_mode") != worker_mode:
        raise RuntimeError("production child worker mode does not match requested worker mode")
    effective_workers = topology.get("effective_rayon_workers")
    if not isinstance(effective_workers, int) or effective_workers < 1:
        raise RuntimeError("production child must report positive effective Rayon workers")
    allowed_parallelism = topology.get("allowed_parallelism")
    if not isinstance(allowed_parallelism, int) or allowed_parallelism < 1:
        raise RuntimeError(
            "production child must report cpuset-aware allowed Rayon parallelism"
        )
    if topology.get("allowed_parallelism_source") != "std::thread::available_parallelism":
        raise RuntimeError(
            "production child must identify std::thread::available_parallelism "
            "as the allowed-parallelism source"
        )
    if effective_workers > allowed_parallelism:
        raise RuntimeError(
            "production child effective Rayon worker count exceeds its allowed CPU set"
        )
    if allowed_parallelism != runner_allowed_cpu_count:
        raise RuntimeError(
            "Python affinity-derived CPU count disagrees with Rust "
            "std::thread::available_parallelism"
        )
    if worker_mode == "default":
        if effective_workers != allowed_parallelism:
            raise RuntimeError(
                "production default worker mode must use the full allowed CPU set"
            )
    elif effective_workers != int(worker_mode):
        raise RuntimeError(
            "production explicit worker mode must use exactly its requested Rayon count"
        )
    if not isinstance(topology.get("process_affinity"), str) or not topology[
        "process_affinity"
    ]:
        raise RuntimeError("production child must report its full allowed affinity")
    if child.get("process_affinity") != topology.get("process_affinity"):
        raise RuntimeError("production child affinity copies disagree")
    affinity_cardinality = topology.get("process_affinity_cardinality")
    if (
        not isinstance(affinity_cardinality, int)
        or isinstance(affinity_cardinality, bool)
        or affinity_cardinality != allowed_parallelism
        or topology.get("affinity_matches_allowed_parallelism") is not True
    ):
        raise RuntimeError(
            "production child affinity cardinality must equal allowed_parallelism"
        )
    pool_start_ids = topology.get("pool_start_worker_ids")
    if (
        not isinstance(pool_start_ids, list)
        or not all(isinstance(worker_id, int) for worker_id in pool_start_ids)
        or topology.get("pool_start_worker_count") != len(pool_start_ids)
        or topology.get("pool_start_worker_count") != effective_workers
        or topology.get("pool_start_evidence_scope")
        != "dedicated_pool_construction_only_not_candidate_work_participation"
    ):
        raise RuntimeError("production child pool-start evidence is malformed")
    if child.get("variant_sequence") != list(variant_sequence):
        raise RuntimeError("production child must preserve JSON A/B sequence ordering")
    if child.get("variant") != expected_variant:
        raise RuntimeError(
            "production child variant does not match the requested product"
        )
    if child.get("runner_pid") != runner_pid:
        raise RuntimeError("production child runner PID is not bound to this runner")
    process_id = child.get("process_id")
    if (
        not isinstance(process_id, int)
        or isinstance(process_id, bool)
        or process_id < 1
        or process_id == runner_pid
    ):
        raise RuntimeError("production child process identity is malformed")
    if child.get("measurement_mode") != mode:
        raise RuntimeError("production child measurement mode mismatch")
    if child.get("candidate_family_scope") != "ranked_unary_candidates_only":
        raise RuntimeError("production child must label its ranked unary candidate scope")
    schedule = child.get("cell_schedule")
    if not isinstance(schedule, dict) or (
        schedule.get("index") != schedule_entry.schedule_index
        or schedule.get("seed") != schedule_seed
        or schedule.get("sha256") != schedule_sha256
        or schedule.get("profile_order") != list(schedule_entry.profile_order)
        or child.get("profile") != schedule_entry.profile
        or child.get("metric") != schedule_entry.metric
        or child.get("input_policy") != schedule_entry.input_policy
        or not isinstance(child.get("workload"), dict)
        or child["workload"].get("name") != schedule_entry.workload
        or worker_mode != schedule_entry.worker_mode
    ):
        raise RuntimeError("production child randomized schedule binding mismatch")
    raw_samples = child.get("raw_samples_ns")
    normalized = child.get("samples_ns")
    loops = child.get("loop_count_per_sample")
    observed_minimum = child.get("sample_region_min_observed_ns")
    calibration = child.get("calibration")
    if (
        not isinstance(raw_samples, list)
        or len(raw_samples) != repetitions
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= MIN_MEASURED_REGION_NS
            for value in raw_samples
        )
        or not isinstance(normalized, list)
        or len(normalized) != repetitions
        or not isinstance(loops, int)
        or isinstance(loops, bool)
        or loops < 1
        or loops > MAX_LOOP_COUNT
    ):
        raise RuntimeError("production child raw timing contract is malformed")
    if not isinstance(calibration, dict):
        raise RuntimeError("production child calibration preflight is missing")
    preflight_samples = calibration.get("preflight_samples_ns")
    preflight_minimum = calibration.get("preflight_min_observed_ns")
    initial_probe = calibration.get("initial_probe_median_ns")
    refinement_rounds = calibration.get("refinement_rounds")
    if (
        child.get("calibration_target_region_ns")
        != CALIBRATION_TARGET_REGION_NS
        or calibration.get("policy")
        != (
            "fixed_loop_count_selected_before_recording_"
            "no_recorded_sample_rescaling_or_filtering"
        )
        or not isinstance(initial_probe, int)
        or isinstance(initial_probe, bool)
        or initial_probe < 1
        or not isinstance(refinement_rounds, int)
        or isinstance(refinement_rounds, bool)
        or not 0 <= refinement_rounds <= MAX_CALIBRATION_REFINEMENTS
        or not isinstance(preflight_samples, list)
        or len(preflight_samples) != CALIBRATION_PREFLIGHT_SAMPLES
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= CALIBRATION_TARGET_REGION_NS
            for value in preflight_samples
        )
        or preflight_minimum != min(preflight_samples)
        or calibration.get("loop_count_limit") != MAX_LOOP_COUNT
    ):
        raise RuntimeError("production child calibration preflight is malformed")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not math.isclose(
            float(value),
            float(raw) / loops,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        for value, raw in zip(normalized, raw_samples, strict=True)
    ):
        raise RuntimeError("production child normalized timings do not match raw/loop")
    if (
        child.get("target_region_ns") != MIN_MEASURED_REGION_NS
        or observed_minimum != min(raw_samples)
        or child.get("sample_region_target_met") is not True
    ):
        raise RuntimeError("production child measured-region floor is not truthful")
    worker_ticks = topology.get("worker_os_cpu_ticks")
    if (
        not isinstance(worker_ticks, list)
        or len(worker_ticks) != effective_workers
        or not all(isinstance(entry, dict) for entry in worker_ticks)
    ):
        raise RuntimeError("production child worker CPU-tick records are malformed")
    tick_ids = [entry.get("worker_id") for entry in worker_ticks]
    invalid_tick_ids = any(
        not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0
        for worker_id in tick_ids
    )
    if (
        tick_ids != pool_start_ids
        or invalid_tick_ids
        or len(set(tick_ids)) != effective_workers
    ):
        raise RuntimeError("production child worker CPU-tick identities are malformed")
    observable_field = topology.get("worker_cpu_ticks_observable")
    positive_field = topology.get("every_effective_worker_positive_work_ticks")
    if not isinstance(observable_field, bool) or not isinstance(positive_field, bool):
        raise RuntimeError(
            "production child worker CPU-tick status flags are malformed"
        )
    ticks_observable = observable_field
    if ticks_observable:
        os_tids = [entry.get("os_tid") for entry in worker_ticks]
        numeric_fields_valid = all(
            isinstance(entry.get(field), int)
            and not isinstance(entry.get(field), bool)
            and entry[field] >= 0
            for entry in worker_ticks
            for field in ("cpu_ticks_before", "cpu_ticks_after", "work_ticks")
        )
        deltas_match = numeric_fields_valid and all(
            entry["cpu_ticks_after"] >= entry["cpu_ticks_before"]
            and entry["work_ticks"]
            == entry["cpu_ticks_after"] - entry["cpu_ticks_before"]
            for entry in worker_ticks
        )
        if (
            any(
                not isinstance(os_tid, int) or isinstance(os_tid, bool) or os_tid < 1
                for os_tid in os_tids
            )
            or len(set(os_tids)) != effective_workers
            or not numeric_fields_valid
            or not deltas_match
        ):
            raise RuntimeError("observable worker CPU-tick evidence is malformed")
        every_positive = all(entry["work_ticks"] > 0 for entry in worker_ticks)
        expected_status = (
            "all_effective_workers_positive"
            if every_positive
            else "observable_but_one_or_more_workers_zero"
        )
        if (
            positive_field is not every_positive
            or topology.get("worker_cpu_tick_status") != expected_status
        ):
            raise RuntimeError("observable worker CPU-tick status is inconsistent")
    else:
        every_positive = False
        if (
            positive_field is not False
            or topology.get("worker_cpu_tick_status") != "portable_unobservable"
            or any(
                entry.get(field) is not None
                for entry in worker_ticks
                for field in (
                    "os_tid",
                    "cpu_ticks_before",
                    "cpu_ticks_after",
                    "work_ticks",
                )
            )
        ):
            raise RuntimeError(
                "unobservable worker CPU ticks require a truthful portable marker"
            )
    if mode == "stable" and not ticks_observable:
        raise RuntimeError(
            "stable production evidence requires observable Linux CPU ticks"
        )
    if mode == "stable" and expected_variant == "candidate" and not every_positive:
        raise RuntimeError(
            "stable candidate evidence requires positive Linux CPU ticks from every worker"
        )
    _canonical_child_environment(child)
    result = child.get("result")
    snapshot = result.get("untimed_snapshot") if isinstance(result, dict) else None
    if not isinstance(snapshot, dict):
        raise RuntimeError("production child lacks its untimed full result snapshot")
    row_count = snapshot.get("row_count")
    max_arity = snapshot.get("max_arity")
    metric_count = snapshot.get("metric_count")
    result_flags = snapshot.get("result_flags")
    metric_ids = snapshot.get("metric_ids")
    combo_indices = snapshot.get("combo_indices")
    ranks = snapshot.get("ranks")
    families = snapshot.get("families")
    candidate_ids = snapshot.get("candidate_ids")
    row_flags = snapshot.get("row_flags")
    value_bits = snapshot.get("metric_value_bits")
    value_text = snapshot.get("metric_value_text")
    value_classes = snapshot.get("metric_value_classes")
    expected_values = (
        row_count * metric_count
        if isinstance(row_count, int) and isinstance(metric_count, int)
        else -1
    )
    expected_combos = (
        row_count * max_arity
        if isinstance(row_count, int) and isinstance(max_arity, int)
        else -1
    )
    def u32_list(values: object, length: int) -> bool:
        return (
            isinstance(values, list)
            and len(values) == length
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 0xFFFF_FFFF
                for value in values
            )
        )

    def u64_list(values: object, length: int) -> bool:
        return (
            isinstance(values, list)
            and len(values) == length
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF
                for value in values
            )
        )
    if (
        not isinstance(row_count, int)
        or row_count < 1
        or not isinstance(max_arity, int)
        or isinstance(max_arity, bool)
        or max_arity < 1
        or not isinstance(metric_count, int)
        or metric_count < 1
        or not isinstance(result_flags, int)
        or isinstance(result_flags, bool)
        or not 0 <= result_flags <= 0xFFFF_FFFF
        or not u32_list(metric_ids, metric_count)
        or metric_ids != [METRIC_IDS.get(str(child.get("metric")))]
        or not u32_list(combo_indices, expected_combos)
        or not u32_list(ranks, row_count)
        or not u32_list(families, row_count)
        or not u64_list(candidate_ids, row_count)
        or not u32_list(row_flags, row_count)
        or not u64_list(value_bits, expected_values)
        or not isinstance(value_text, list)
        or len(value_text) != expected_values
        or not isinstance(value_classes, list)
        or len(value_classes) != expected_values
        or not _snapshot_values_are_self_consistent(snapshot)
    ):
        raise RuntimeError("production child full result snapshot is malformed")
    if snapshot.get("result_dtype") != (
        "f32" if child.get("profile") == "fp32" else "f64"
    ):
        raise RuntimeError("production child snapshot dtype/profile mismatch")
    structural_digest, metric_digest = _snapshot_digests(snapshot)
    if (
        not isinstance(result, dict)
        or result.get("rows_written") != row_count
        or result.get("candidate_digest") != structural_digest
        or result.get("visible_score_bits") != metric_digest
        or result.get("digest_scope")
        != "all_visible_result_metadata_structural_arrays_and_metric_bits"
    ):
        raise RuntimeError("timed result digest does not authenticate untimed snapshot")
    _canonical_child_device(child)
    clock_power = child.get("clock_and_power_state")
    if (
        child.get("clock_and_power_capture_point")
        != "before and after all timed benchmark regions"
        or not isinstance(clock_power, dict)
        or not isinstance(clock_power.get("before"), dict)
        or not isinstance(clock_power.get("after"), dict)
    ):
        raise RuntimeError("production child must capture before/after CPU clock state")


def _write_manifest(
    path: Path,
    *,
    output: FileIdentity,
    product: RepositoryIdentity,
    variant: str,
    ab_block: int,
    variant_sequence: tuple[str, ...],
) -> None:
    manifest_parent = path.parent.resolve()
    try:
        relative_output = output.path.resolve().relative_to(manifest_parent)
    except ValueError as error:
        raise ValueError(
            "production manifest and aggregate must share one portable evidence root"
        ) from error
    payload = {
        "schema": NATIVE_EVIDENCE_SCHEMA,
        "status": "validated",
        "arithmetic_claims_valid": True,
        "source_commit": product.commit,
        "artifacts": [
            {
                "variant": variant,
                "backend": "core",
                "kind": "core_production_executor",
                "path": relative_output.as_posix(),
                "relative_path": relative_output.as_posix(),
                "sha256": output.sha256,
                "schedule": {
                    "variant": variant,
                    "ab_block": ab_block,
                    "variant_sequence": list(variant_sequence),
                    "process_isolation": PROCESS_ISOLATION,
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    runner_pid = os.getpid()
    profiles, metrics, workloads, policies, workers = _validate_arguments(args)
    git = _git_identity()
    product = _repository_identity(args.product_source_root)
    harness = _repository_identity(args.harness_source_root)
    _validate_variant_product_binding(args.variant, product.commit)
    if args.expected_product_commit and product.commit != args.expected_product_commit.lower():
        raise ValueError("product checkout does not match --expected-product-commit")
    if args.expected_harness_commit and harness.commit != args.expected_harness_commit.lower():
        raise ValueError("harness checkout does not match --expected-harness-commit")
    source = _tracked_source_identity(harness.root, HARNESS_SOURCE)
    runner_source = _tracked_source_identity(harness.root, HARNESS_RUNNER)
    product_rlib = _file_identity(args.product_rlib)
    orchestrator_rlib = _file_identity(args.orchestrator_rlib)
    types_rlib = _file_identity(args.types_rlib)
    rayon_rlib = _file_identity(args.rayon_rlib)
    wheel = _file_identity(args.wheel)
    if product_rlib.path.suffix != ".rlib" or "gafime_cpu" not in product_rlib.path.name:
        raise ValueError("--product-rlib must name the exact gafime_cpu rlib")
    if "gafime_orchestrator" not in orchestrator_rlib.path.name:
        raise ValueError("--orchestrator-rlib must name the exact gafime_orchestrator rlib")
    if "gafime_types" not in types_rlib.path.name:
        raise ValueError("--types-rlib must name the exact gafime_types rlib")
    if "rayon" not in rayon_rlib.path.name:
        raise ValueError("--rayon-rlib must name the exact Rayon rlib")
    if wheel.path.suffix != ".whl":
        raise ValueError("--wheel must name the exact Core wheel")
    dependency_dirs = tuple(
        dict.fromkeys(
            directory.resolve(strict=True)
            for directory in (
                [product_rlib.path.parent, orchestrator_rlib.path.parent, types_rlib.path.parent, rayon_rlib.path.parent]
                + list(args.dependency_dir)
            )
        )
    )
    if any(not directory.is_dir() for directory in dependency_dirs):
        raise ValueError("--dependency-dir must name an existing directory")
    binary = args.binary.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("--output already exists; refuse to overwrite prior evidence")
    if args.manifest_output and args.manifest_output.resolve().exists():
        raise ValueError(
            "--manifest-output already exists; refuse to overwrite prior evidence"
        )
    raw_dir = output.parent / f"{output.stem}.raw"
    if raw_dir.exists():
        raise ValueError("raw child artifact directory already exists; refuse to mix evidence")
    binary.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = _compiler_command(
        rustup=args.rustup,
        toolchain=args.toolchain,
        source=source.path,
        product_rlib=product_rlib.path,
        orchestrator_rlib=orchestrator_rlib.path,
        types_rlib=types_rlib.path,
        rayon_rlib=rayon_rlib.path,
        dependency_dirs=dependency_dirs,
        binary=binary,
    )
    compile_environment = _compiler_environment(
        source,
        runner_source,
        product_rlib,
        orchestrator_rlib,
        types_rlib,
        rayon_rlib,
        compiler,
    )
    _run(compiler, env=compile_environment)
    binary_identity = _file_identity(binary)
    rustc_version = _one_line([args.rustup, "run", args.toolchain, "rustc", "--version"])
    try:
        linker_version = _one_line(["cc", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        linker_version = _one_line(["ld", "--version"])
    base_environment = _benchmark_environment()
    raw_dir.mkdir()
    variant_sequence = _parse_csv(
        args.variant_sequence,
        allowed={"baseline", "candidate"},
        label="--variant-sequence",
    )
    scaling_plan = _scaling_execution_plan(workers)
    schedule, schedule_seed, schedule_sha256, profile_order_counts = _cell_schedule(
        profiles=profiles,
        metrics=metrics,
        workloads=workloads,
        policies=policies,
        worker_modes=scaling_plan.executed_worker_modes,
        seed=args.seed,
        ab_block=args.ab_block,
    )
    records: list[dict[str, object]] = []
    child_artifacts: list[dict[str, object]] = []
    observed_child_pids: set[int] = set()
    for entry in schedule:
        index = entry.schedule_index
        policy = entry.input_policy
        workload = entry.workload
        metric = entry.metric
        profile = entry.profile
        worker_mode = entry.worker_mode
        child_output = raw_dir / (
            f"{index:04d}-{policy}-{workload}-{metric}-{profile}-workers-{worker_mode}.json"
        )
        invocation_id = secrets.token_hex(16)
        child_environment = _child_environment(
            base=base_environment,
            product=product,
            harness=harness,
            source=source,
            runner_source=runner_source,
            product_rlib=product_rlib,
            orchestrator_rlib=orchestrator_rlib,
            types_rlib=types_rlib,
            rayon_rlib=rayon_rlib,
            wheel=wheel,
            binary=binary_identity,
            output=child_output,
            profile=profile,
            metric=metric,
            workload=workload,
            policy=policy,
            workers=worker_mode,
            warmups=args.warmups,
            repetitions=args.repetitions,
            variant=args.variant,
            ab_block=args.ab_block,
            variant_sequence=variant_sequence,
            runner_invocation_id=invocation_id,
            runner_pid=runner_pid,
            mode=args.mode,
            schedule_index=entry.schedule_index,
            schedule_seed=schedule_seed,
            schedule_sha256=schedule_sha256,
            profile_order=entry.profile_order,
        )
        _run([str(binary)], env=child_environment, capture_output=False)
        child = _load_child(child_output)
        _validate_child_contract(
            child,
            expected_variant=args.variant,
            worker_mode=worker_mode,
            variant_sequence=variant_sequence,
            runner_pid=runner_pid,
            mode=args.mode,
            schedule_entry=entry,
            schedule_seed=schedule_seed,
            schedule_sha256=schedule_sha256,
            repetitions=args.repetitions,
            runner_allowed_cpu_count=scaling_plan.allowed_cpu_count,
        )
        child_identity = _file_identity(child_output)
        process_id = child["process_id"]
        assert isinstance(process_id, int)
        if process_id in observed_child_pids:
            raise RuntimeError("fresh production child PID was reused within one runner")
        observed_child_pids.add(process_id)
        _record_statistics(child, seed=args.seed + index, resamples=args.bootstrap_resamples)
        raw_identity = _identity_json(child_identity)
        raw_identity["relative_path"] = child_output.relative_to(output.parent).as_posix()
        child["raw_child_artifact"] = raw_identity
        child["runner_invocation_id"] = invocation_id
        records.append(child)
        child_artifacts.append(
            {
                "index": index,
                "profile": profile,
                "metric": metric,
                "workload": workload,
                "input_policy": policy,
                "worker_mode": worker_mode,
                "profile_order": list(entry.profile_order),
                "profile_order_ordinal": entry.profile_order_ordinal,
                "schedule_sha256": schedule_sha256,
                **raw_identity,
            }
        )
    raw_measurement_claim_ready = all(
        record.get("claim_ready") is True for record in records
    )
    worker_topology_claim_ready = all(
        isinstance(record.get("execution_topology"), dict)
        and record["execution_topology"].get("worker_cpu_ticks_observable") is True
        and record["execution_topology"].get(
            "every_effective_worker_positive_work_ticks"
        )
        is True
        for record in records
    )
    canonical_environments = [_canonical_child_environment(record) for record in records]
    if not canonical_environments or any(
        environment != canonical_environments[0]
        for environment in canonical_environments[1:]
    ):
        raise RuntimeError(
            "fresh production children observed inconsistent canonical environments"
        )
    canonical_device = _canonical_common_child_device(records)
    thread_scaling_tables = _thread_scaling_tables(records)
    primary_record_indexes = [
        index
        for index, record in enumerate(records)
        if isinstance(record.get("execution_topology"), dict)
        and record["execution_topology"].get("measurement_role")
        == "primary_default_worker_production_result"
    ]
    scaling_record_indexes = [
        index
        for index, record in enumerate(records)
        if isinstance(record.get("execution_topology"), dict)
        and record["execution_topology"].get("measurement_role")
        == "thread_scaling_diagnostic"
    ]
    observed_affinities = sorted(
        {
            str(record["execution_topology"].get("process_affinity"))
            for record in records
            if isinstance(record.get("execution_topology"), dict)
            and record["execution_topology"].get("process_affinity")
        }
    )
    if len(observed_affinities) != 1:
        raise RuntimeError(
            "fresh production children observed inconsistent allowed CPU affinity; "
            "measurement is contaminated"
        )
    child_clock_states = [
        record.get("clock_and_power_state")
        for record in records
        if isinstance(record.get("clock_and_power_state"), dict)
    ]
    if len(child_clock_states) != len(records):
        raise RuntimeError("every production child must retain clock/power provenance")
    static_clock_views = [_static_clock_power_view(state) for state in child_clock_states]
    static_clock_json = {
        json.dumps(view, sort_keys=True, separators=(",", ":"))
        for view in static_clock_views
    }
    for view in static_clock_views:
        if view["before"] != view["after"]:
            raise RuntimeError(
                "static CPU clock/power policy changed during a production child; "
                "reject contaminated evidence"
            )
    if len(static_clock_json) != 1:
        raise RuntimeError(
            "fresh production children observed inconsistent static clock/power policy; "
            "reject contaminated evidence"
        )
    canonical_clock_power_state = json.loads(next(iter(static_clock_json)))
    output_payload: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "pass" if raw_measurement_claim_ready else "informational",
        # This is intentionally raw-cell readiness only.  A global perf13
        # aggregation must independently establish both A/B and B/A blocks
        # before a comparative claim may be made.
        "performance_claim_ready": False,
        "raw_measurement_claim_ready": raw_measurement_claim_ready,
        "worker_topology_claim_ready": worker_topology_claim_ready,
        "worker_topology_claim_scope": (
            "required for the repaired candidate in stable mode; the frozen "
            "pre-repair baseline may truthfully retain idle Rayon workers"
        ),
        "comparative_performance_claim_ready": False,
        "comparative_claim_boundary": (
            "raw child integrity is separate from comparative performance "
            "readiness; perf13 must verify paired A/B and B/A blocks"
        ),
        "backend": "core",
        "measurement_mode": args.mode,
        "profiles": list(profiles),
        "metrics": list(metrics),
        "workloads": list(workloads),
        "input_policies": list(policies),
        "worker_modes": list(workers),
        "scaling_execution_plan": {
            "allowed_cpu_count": scaling_plan.allowed_cpu_count,
            "requested_worker_modes": list(scaling_plan.requested_worker_modes),
            "executed_worker_modes": list(scaling_plan.executed_worker_modes),
            "skipped_worker_modes": list(scaling_plan.skipped_worker_modes),
            "policy": (
                "explicit 1/2/4 diagnostics never oversubscribe the observed "
                "CPU set; unavailable labels are skipped with a bound reason"
            ),
        },
        "measurement_scope": "production_precision_compute_backend",
        "candidate_family_scope": "ranked_unary_candidates_only",
        "result_snapshot_parity_policy": SNAPSHOT_PARITY_POLICY,
        "decomposition_boundaries": {
            "ingest_conversion": (
                "not present in the timed resident executor; typed matrix "
                "construction is recorded as untimed setup"
            ),
            "candidate_materialization": (
                "fused within the timed ranked PrecisionComputeBackend execution"
            ),
        },
        "claim_boundary": {
            "eligible_for_core_production_throughput": True,
            "excludes": [
                "supplemental_single_core_leaf_kernel_diagnostic",
                "direct_metric_kernel_only",
            ],
        },
        "execution_topology": {
            "candidate_parallelism": (
                "rayon_candidate_level"
                if args.variant == "candidate"
                else "frozen_pre_repair_serial_candidate_loop"
            ),
            "semantic_candidate_participation_guard": (
                "cfg_test_precision_executor_parallelism_contract"
                if args.variant == "candidate"
                else "not_applicable_frozen_pre_repair_serial_baseline"
            ),
            "required_scaling_modes": list(workers),
            "process_isolation": PROCESS_ISOLATION,
            "primary_default_worker_record_indexes": primary_record_indexes,
            "thread_scaling_record_indexes": scaling_record_indexes,
        },
        "thread_scaling_tables": thread_scaling_tables,
        "primary_result_scope": (
            "default-worker ranked unary production executor cells only; explicit "
            "worker-count cells are separate scaling diagnostics"
        ),
        "source_commit": product.commit,
        "product_source_commit": product.commit,
        "product_source_tree": product.tree,
        "product_source_tree_state": {"status": "clean"},
        "harness_source_commit": harness.commit,
        "harness_source_tree": harness.tree,
        "harness_source_tree_state": {"status": "clean"},
        "harness_source_blob": {
            "relative_path": source.relative_path.as_posix(),
            "source_sha256": source.sha256,
            "current_git_blob": source.git_blob,
            "head_git_blob": source.git_blob,
        },
        "harness_runner_blob": {
            "relative_path": runner_source.relative_path.as_posix(),
            "source_sha256": runner_source.sha256,
            "current_git_blob": runner_source.git_blob,
            "head_git_blob": runner_source.git_blob,
        },
        "source_tree_state": {"status": "clean"},
        "warmups": args.warmups,
        "repeats": args.repetitions,
        "bootstrap_resamples": args.bootstrap_resamples,
        "variant": args.variant,
        "ab_block": args.ab_block,
        "variant_sequence": list(variant_sequence),
        "runner_pid": runner_pid,
        "child_process_ids": sorted(observed_child_pids),
        "cell_schedule": {
            "algorithm": "seeded_balanced_profile_orders_v1",
            "seed": schedule_seed,
            "sha256": schedule_sha256,
            "ab_block": args.ab_block,
            "entry_count": len(schedule),
            "profile_order_counts": profile_order_counts,
            "entries": [
                {
                    "schedule_index": entry.schedule_index,
                    "input_policy": entry.input_policy,
                    "workload": entry.workload,
                    "metric": entry.metric,
                    "profile": entry.profile,
                    "worker_mode": entry.worker_mode,
                    "profile_order": list(entry.profile_order),
                    "profile_order_ordinal": entry.profile_order_ordinal,
                }
                for entry in schedule
            ],
        },
        "process_isolation": PROCESS_ISOLATION,
        "process_affinity": observed_affinities[0],
        "observed_process_affinities": observed_affinities,
        "device": canonical_device,
        "device_scope": (
            "canonical identical CPU identity and logical/physical topology "
            "reported by every fresh child"
        ),
        "environment": canonical_environments[0],
        "environment_scope": (
            "canonical actual child environment shared by every fresh process; "
            "RAYON_NUM_THREADS is explicitly scrubbed"
        ),
        "command_line": [str(binary)],
        "clock": "std::time::Instant monotonic clock",
        "clock_and_power_capture_point": "before and after all timed benchmark regions",
        "clock_and_power_state": {
            "before": canonical_clock_power_state["before"],
            "after": canonical_clock_power_state["after"],
        },
        "per_child_clock_and_power_state": child_clock_states,
        "compiler": {
            "rustc": rustc_version,
            "linker": linker_version,
            "toolchain": args.toolchain,
            "edition": "2021",
            "codegen_flags": [
                "-Copt-level=3",
                "-Ccodegen-units=1",
                "-Clto=fat",
                "-Cembed-bitcode=yes",
            ],
            "command_argv": compiler,
            "command_sha256": hashlib.sha256(
                json.dumps(compiler, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            ).hexdigest(),
        },
        "provenance": {
            "benchmark_source": _identity_json(_file_identity(source.path)),
            "harness_source": _identity_json(_file_identity(source.path)),
            "harness_runner": _identity_json(_file_identity(runner_source.path)),
            "benchmark_binary": _identity_json(binary_identity),
            "wheel": _identity_json(wheel),
            "product_rlib": _identity_json(product_rlib),
            "orchestrator_rlib": _identity_json(orchestrator_rlib),
            "types_rlib": _identity_json(types_rlib),
            "rayon_rlib": _identity_json(rayon_rlib),
            "python_executable": _identity_json(_file_identity(Path(sys.executable))),
        },
        "git_provenance": {
            **_git_record(git),
            "repositories": [
                {
                    "root": str(repository.root),
                    "commit": repository.commit,
                    "tree": repository.tree,
                    "git_dir": str(repository.git_dir),
                    "git_common_dir": str(repository.git_common_dir),
                    "clean_tree_verified": True,
                }
                for repository in (product, harness)
            ],
        },
        "records": records,
        "raw_child_artifacts": child_artifacts,
    }
    output.write_text(json.dumps(output_payload, sort_keys=True) + "\n", encoding="utf-8")
    if args.manifest_output:
        _write_manifest(
            args.manifest_output.resolve(),
            output=_file_identity(output),
            product=product,
            variant=args.variant,
            ab_block=args.ab_block,
            variant_sequence=variant_sequence,
        )
    if args.mode == "stable" and not raw_measurement_claim_ready:
        raise RuntimeError("stable production benchmark mode requires release-ready child cells")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Core production benchmark runner failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
