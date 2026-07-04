from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "gafime.telemetry.v0.5.0-rc1"

SPAN_KEYS = (
    "e2e_total",
    "openml_load_preprocess",
    "python_orchestration_gil",
    "gafime_planning_session_report",
    "gafime_rust_orchestrator",
    "gafime_cpp_core",
    "gafime_cpu_execution",
    "gafime_gpu_execution",
    "gpu_kernel_launch",
    "host_device_transfer",
    "h2d_transfer",
    "d2h_transfer",
    "result_ranking",
    "result_materialization",
    "gafime_to_downstream_transfer",
    "downstream_fit",
)

DEEP_TELEMETRY_SPANS = (
    "e2e_total",
    "python_orchestration_gil",
    "gafime_rust_orchestrator",
    "gafime_cpu_execution",
    "gafime_gpu_execution",
    "host_device_transfer",
    "h2d_transfer",
    "d2h_transfer",
    "gpu_kernel_launch",
    "result_ranking",
    "result_materialization",
    "gafime_to_downstream_transfer",
    "downstream_fit",
)

COUNTER_KEYS = (
    "pybind_calls",
    "ctypes_calls",
    "h2d_bytes",
    "d2h_bytes",
    "model_transfer_bytes",
    "native_allocations",
    "gpu_allocations",
    "peak_rss_bytes",
    "peak_vram_bytes",
)

CSV_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_branch",
    "dataset_source",
    "openml_id",
    "dataset_name",
    "backend",
    "status",
    "e2e_total_ns",
    "gafime_rust_orchestrator_ns",
    "gafime_cpp_core_ns",
    "gafime_cpu_execution_ns",
    "gafime_gpu_execution_ns",
    "downstream_fit_ns",
    "gafime_to_downstream_transfer_ns",
    "python_orchestration_gil_ns",
    "gpu_kernel_launch_ns",
    "host_device_transfer_ns",
    "h2d_transfer_ns",
    "d2h_transfer_ns",
    "result_ranking_ns",
    "result_materialization_ns",
    "baseline_score",
    "gafime_score",
    "training_time_reduction",
    "structural_recovery",
    "artifact_path",
)


def monotonic_ns() -> int:
    return time.perf_counter_ns()


@contextmanager
def span(record: dict[str, Any], name: str) -> Iterator[None]:
    if name not in SPAN_KEYS:
        raise ValueError(f"Unknown telemetry span: {name}")
    start = monotonic_ns()
    try:
        yield
    finally:
        elapsed = monotonic_ns() - start
        current = record.setdefault("spans_ns", {}).get(name)
        record["spans_ns"][name] = elapsed if current is None else int(current) + elapsed


def ensure_deep_telemetry_spans(record: dict[str, Any], *, fill: int | None = 0) -> dict[str, Any]:
    spans = record.setdefault("spans_ns", {})
    for key in SPAN_KEYS:
        spans.setdefault(key, None)
    for key in DEEP_TELEMETRY_SPANS:
        if spans.get(key) is None:
            spans[key] = fill
    return record


def finalize_e2e_and_python_overhead(record: dict[str, Any], e2e_start_ns: int) -> dict[str, Any]:
    spans = record.setdefault("spans_ns", {})
    e2e_total = monotonic_ns() - e2e_start_ns
    spans["e2e_total"] = e2e_total
    accounted = sum(
        int(value)
        for key, value in spans.items()
        if key != "e2e_total" and isinstance(value, int)
    )
    spans["python_orchestration_gil"] = max(0, e2e_total - accounted)
    ensure_deep_telemetry_spans(record)
    return record


def record_kernel(
    record: dict[str, Any],
    *,
    backend: str,
    name: str,
    duration_ns: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    arity: int | None = None,
    graph_replay: bool | None = None,
) -> dict[str, Any]:
    event = {
        "backend": backend,
        "name": name,
        "duration_ns": duration_ns,
        "rows": rows,
        "cols": cols,
        "arity": arity,
        "graph_replay": graph_replay,
    }
    record.setdefault("gpu_kernels", []).append(event)
    if duration_ns is not None:
        spans = record.setdefault("spans_ns", {})
        current = spans.get("gpu_kernel_launch")
        spans["gpu_kernel_launch"] = int(duration_ns) if current is None else int(current) + int(duration_ns)
    return record


def validate_deep_telemetry(record: dict[str, Any]) -> None:
    missing = [key for key in DEEP_TELEMETRY_SPANS if key not in record.get("spans_ns", {})]
    if missing:
        raise ValueError(f"Missing deep telemetry spans: {missing}")
    non_numeric = [
        key
        for key in DEEP_TELEMETRY_SPANS
        if not isinstance(record.get("spans_ns", {}).get(key), int)
    ]
    if non_numeric:
        raise ValueError(f"Deep telemetry spans must be integer nanoseconds: {non_numeric}")


def new_record(
    *,
    worktree: str | os.PathLike[str],
    dataset: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    package_names: tuple[str, ...] = ("gafime", "numpy", "scikit-learn", "openml"),
) -> dict[str, Any]:
    worktree_path = Path(worktree)
    run = git_metadata(worktree_path)
    run["run_id"] = uuid.uuid4().hex
    run["timestamp_utc"] = _utc_timestamp()
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "environment": environment_metadata(package_names=package_names),
        "dataset": _default_dataset() | (dataset or {}),
        "config": _default_config() | (config or {}),
        "spans_ns": {key: None for key in SPAN_KEYS},
        "counters": {key: None for key in COUNTER_KEYS},
        "gpu_kernels": [],
        "results": {
            "status": "pass",
            "exception": None,
            "baseline_score": None,
            "gafime_score": None,
            "training_time_reduction": None,
            "structural_recovery": None,
            "top_candidates": [],
        },
        "artifacts": [],
    }


def git_metadata(worktree: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(worktree)
    return {
        "git_commit": _git(root, "rev-parse", "HEAD") or None,
        "git_branch": _git(root, "branch", "--show-current") or None,
        "worktree": str(root.resolve()),
        "dirty": bool(_git(root, "status", "--short")),
    }


def environment_metadata(
    *,
    package_names: tuple[str, ...] = ("gafime", "numpy", "scikit-learn", "openml"),
) -> dict[str, Any]:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count() or 0,
        "cpu_threads": os.cpu_count() or 0,
        "ram_bytes": _linux_ram_bytes(),
        "gpu": gpu_metadata(),
        "compiler": None,
        "openmp": None,
        "python": platform.python_version(),
        "packages": package_versions(package_names),
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_metadata() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        output = _run(["nvidia-smi", "-L"])
        for line in output.splitlines():
            if line.strip():
                gpus.append({
                    "name": line.strip(),
                    "backend": "cuda",
                    "arch": None,
                    "driver": None,
                    "runtime": None,
                })
    return gpus


def write_run(
    record: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    csv_name: str = "index.csv",
) -> tuple[Path, Path]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported telemetry schema: {record.get('schema_version')!r}")
    ensure_deep_telemetry_spans(record)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(record["run"]["run_id"])
    json_path = out / f"{run_id}.json"
    record.setdefault("artifacts", []).append(str(json_path))
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = out / csv_name
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(csv_row(record, artifact_path=json_path))
    return json_path, csv_path


def csv_row(record: dict[str, Any], *, artifact_path: str | os.PathLike[str]) -> dict[str, Any]:
    run = record.get("run", {})
    dataset = record.get("dataset", {})
    config = record.get("config", {})
    spans = record.get("spans_ns", {})
    results = record.get("results", {})
    return {
        "run_id": run.get("run_id"),
        "timestamp_utc": run.get("timestamp_utc"),
        "git_commit": run.get("git_commit"),
        "git_branch": run.get("git_branch"),
        "dataset_source": dataset.get("source"),
        "openml_id": dataset.get("openml_id"),
        "dataset_name": dataset.get("name"),
        "backend": config.get("backend"),
        "status": results.get("status"),
        "e2e_total_ns": spans.get("e2e_total"),
        "gafime_rust_orchestrator_ns": spans.get("gafime_rust_orchestrator"),
        "gafime_cpp_core_ns": spans.get("gafime_cpp_core"),
        "gafime_cpu_execution_ns": spans.get("gafime_cpu_execution"),
        "gafime_gpu_execution_ns": spans.get("gafime_gpu_execution"),
        "downstream_fit_ns": spans.get("downstream_fit"),
        "gafime_to_downstream_transfer_ns": spans.get("gafime_to_downstream_transfer"),
        "python_orchestration_gil_ns": spans.get("python_orchestration_gil"),
        "gpu_kernel_launch_ns": spans.get("gpu_kernel_launch"),
        "host_device_transfer_ns": spans.get("host_device_transfer"),
        "h2d_transfer_ns": spans.get("h2d_transfer"),
        "d2h_transfer_ns": spans.get("d2h_transfer"),
        "result_ranking_ns": spans.get("result_ranking"),
        "result_materialization_ns": spans.get("result_materialization"),
        "baseline_score": results.get("baseline_score"),
        "gafime_score": results.get("gafime_score"),
        "training_time_reduction": results.get("training_time_reduction"),
        "structural_recovery": results.get("structural_recovery"),
        "artifact_path": str(artifact_path),
    }


def mark_failed(record: dict[str, Any], exc: BaseException) -> None:
    record.setdefault("results", {})["status"] = "fail"
    record["results"]["exception"] = f"{type(exc).__name__}: {exc}"


def _default_dataset() -> dict[str, Any]:
    return {
        "source": "synthetic",
        "openml_id": None,
        "name": None,
        "version": None,
        "rows": 0,
        "features": 0,
        "target_type": None,
        "seed": None,
        "split_policy": None,
    }


def _default_config() -> dict[str, Any]:
    return {
        "backend": None,
        "metric_names": [],
        "gafime": {},
        "downstream_model": {},
    }


def _git(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root)


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _linux_ram_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
