"""Opt-in installed-package/native-ABI collision regression. Never a default test.

Run --help without GPU imports. The controller launches one bounded isolated
Python subprocess; only that subprocess imports GAFIME or opens the payload.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
CASES = ("graph-abi10", "graph-abi11", "graph-both", "eager-both", "foreign-only")
KNOWN_UNFIXED_CUDA_RC2 = (
    "ab4051b0de2d363c472ac3f02d35104927636b2404fc43a0832901cb4a7c08c0"
)
SOURCE_FILES = (
    "src/common/gpu_execution_gate.hpp",
    "src/cuda/precision_launcher.cu",
    "src/rocm/launcher.hip",
    "tests/gpu/abi_consumers/abi_1_0_c_consumer.c",
    "tests/gpu/abi_consumers/abi_1_1_c_consumer.c",
    "tests/gpu/abi_consumers/abi_dynamic_load.h",
    "tests/gpu/abi_consumers/CMakeLists.txt",
    "tests/gpu/execution_coordination_regression.py",
)
LOG_LIMIT_BYTES = 16 * 1024 * 1024
COMPLETION_LIMIT_BYTES = 64 * 1024
COMPLETION_FILE = "completion.json"


@dataclass
class CallProgress:
    calls: int = 0
    first_started_at: float | None = None
    last_finished_at: float | None = None
    call_seconds: float = 0.0


def run_until_deadline(
    operation, deadline, stop, progress, *, clock=time.monotonic, pause=None
):
    """Timed control shared by the real primary/foreign loops and host tests."""
    while not stop.is_set():
        began = clock()
        if began >= deadline:
            return
        if progress.first_started_at is None:
            progress.first_started_at = began
        try:
            operation()
        finally:
            finished = clock()
            progress.calls += 1
            progress.last_finished_at = finished
            progress.call_seconds += finished - began
        if pause is not None:
            pause()


def progress_summary(progress, started_at, deadline):
    requested = deadline - started_at
    # Allow bounded scheduling/loop overhead, not an arbitrary early finish.
    slack = min(0.25, requested * 0.05)
    first = progress.first_started_at
    last = progress.last_finished_at
    covered = (
        progress.calls >= 2
        and first is not None
        and last is not None
        and first <= started_at + slack
        and last >= deadline - slack
    )
    return {
        "calls": progress.calls,
        "first_call_offset_seconds": None if first is None else first - started_at,
        "last_call_offset_seconds": None if last is None else last - started_at,
        "active_span_seconds": 0.0 if first is None or last is None else last - first,
        "call_seconds": progress.call_seconds,
        "coverage_slack_seconds": slack,
        "covered_interval": covered,
    }


def config_sha256(config):
    encoded = json.dumps(
        {key: value for key, value in config.items() if key != "config_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def selected_abis(case):
    if case == "graph-abi10":
        return ["abi10"]
    if case == "graph-abi11":
        return ["abi11"]
    return ["abi10", "abi11"]


def write_completion(output, config, identity, result):
    """Exactly one bounded record, separate from interleaved C/Python stdout."""
    record = {
        "version": 1,
        "run_id": config["run_id"],
        "config_sha256": config["config_sha256"],
        "identity": identity,
        "result": result,
    }
    encoded = (json.dumps(record, allow_nan=False) + "\n").encode()
    if len(encoded) > COMPLETION_LIMIT_BYTES:
        raise ValueError("completion record exceeds size limit")
    with (output / COMPLETION_FILE).open("xb") as stream:
        stream.write(encoded)


def validate_completion(output, config, child_pid, controller_elapsed):
    """Fail closed on absent, repeated, malformed or unrelated child evidence."""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate completion key: {key}")
            result[key] = value
        return result

    def finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def digest(value):
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def check_progress(value, calls, elapsed, name):
        require(isinstance(value, dict), f"missing {name} progress")
        require(type(calls) is int and calls >= 2, f"invalid {name} call count")
        require(
            type(value["calls"]) is int and value["calls"] == calls,
            f"inconsistent {name} call count",
        )
        first, last = (
            value["first_call_offset_seconds"],
            value["last_call_offset_seconds"],
        )
        total = value["call_seconds"]
        require(
            all(finite_number(item) for item in (first, last, total))
            and 0 <= first <= last <= elapsed
            and 0 <= total <= last - first + 1e-8,
            f"invalid {name} timing",
        )
        recomputed = progress_summary(
            CallProgress(calls, first, last, total), 0.0, config["seconds"]
        )
        require(set(value) == set(recomputed), f"malformed {name} progress")
        require(
            value["covered_interval"] is True and recomputed["covered_interval"],
            f"incomplete {name} interval",
        )
        for key in ("active_span_seconds", "coverage_slack_seconds"):
            require(
                finite_number(value[key])
                and math.isclose(
                    value[key], recomputed[key], rel_tol=1e-9, abs_tol=1e-8
                ),
                f"inconsistent {name} {key}",
            )

    try:
        path = output / COMPLETION_FILE
        require(
            path.is_file() and not path.is_symlink(), "missing regular completion file"
        )
        with path.open("rb") as stream:
            encoded = stream.read(COMPLETION_LIMIT_BYTES + 1)
        require(
            len(encoded) <= COMPLETION_LIMIT_BYTES,
            "completion record exceeds size limit",
        )
        record = json.loads(encoded, object_pairs_hook=unique_object)
        require(
            isinstance(record, dict)
            and set(record)
            == {"version", "run_id", "config_sha256", "identity", "result"},
            "malformed completion envelope",
        )
        require(
            type(record["version"]) is int and record["version"] == 1,
            "unknown completion version",
        )
        require(
            record["run_id"] == config["run_id"], "completion run identity mismatch"
        )
        require(
            record["config_sha256"] == config["config_sha256"] == config_sha256(config),
            "completion configuration hash mismatch",
        )
        identity = record["identity"]
        require(
            isinstance(identity, dict)
            and set(identity)
            == {"event", "pid", "python", "core", "core_sha256", "config"},
            "malformed child identity",
        )
        require(identity["event"] == "identity", "missing child identity event")
        require(
            type(identity["pid"]) is int and identity["pid"] == child_pid,
            "child pid mismatch",
        )
        require(identity["python"] == config["python"], "child interpreter mismatch")
        require(
            identity["config"] == config
            and config_sha256(identity["config"]) == config["config_sha256"],
            "child configuration mismatch",
        )
        core = Path(identity["core"])
        require(core.is_absolute() and core.is_file(), "missing child Core")
        require(
            digest(identity["core_sha256"]) and sha256(core) == identity["core_sha256"],
            "child Core hash mismatch",
        )
        result = record["result"]
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "event",
                "status",
                "requested_seconds",
                "elapsed_seconds",
                "primary_calls",
                "primary_progress",
                "foreign_calls",
                "foreign_progress",
                "incomplete_interval_workers",
                "foreign_failures",
                "primary_error",
                "parity_sha256",
                "excluded_report_fields",
            },
            "malformed child result",
        )
        require(
            result["event"] == "result" and result["status"] == "pass",
            "child workload did not pass",
        )
        require(
            type(result["requested_seconds"]) is int
            and result["requested_seconds"] == config["seconds"],
            "child interval mismatch",
        )
        elapsed = result["elapsed_seconds"]
        require(
            finite_number(elapsed)
            and config["seconds"]
            <= elapsed
            <= min(config["timeout"], controller_elapsed),
            "invalid workload elapsed time",
        )
        require(
            result["incomplete_interval_workers"] == []
            and result["foreign_failures"] == []
            and result["primary_error"] is None,
            "child workload reports failures",
        )
        workers = {
            f"{abi}-{worker}"
            for abi in selected_abis(config["case"])
            for worker in range(config["workers"])
        }
        require(
            isinstance(result["foreign_calls"], dict)
            and isinstance(result["foreign_progress"], dict)
            and set(result["foreign_calls"]) == workers
            and set(result["foreign_progress"]) == workers,
            "child ABI worker set mismatch",
        )
        for worker in workers:
            check_progress(
                result["foreign_progress"][worker],
                result["foreign_calls"][worker],
                elapsed,
                worker,
            )
        if config["case"] == "foreign-only":
            require(
                type(result["primary_calls"]) is int
                and result["primary_calls"] == 0
                and result["primary_progress"] is None
                and result["parity_sha256"] is None,
                "unexpected primary workload evidence",
            )
        else:
            check_progress(
                result["primary_progress"], result["primary_calls"], elapsed, "primary"
            )
            require(digest(result["parity_sha256"]), "missing full-report parity hash")
        require(
            result["excluded_report_fields"] == ["backend.memory_free_mb"],
            "unexpected report comparison exclusions",
        )
        return {
            "valid": True,
            "error": None,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as error:
        return {"valid": False, "error": str(error), "sha256": None}


def terminate_and_reap(process, deadline, *, clock=time.monotonic):
    """Never wait without a timeout, even after requesting termination."""
    error = None
    try:
        process.kill()
    except OSError as failure:
        error = str(failure)
    try:
        process.wait(timeout=max(0.0, deadline - clock()))
    except subprocess.TimeoutExpired:
        return {"reaped": False, "reap_timed_out": True, "kill_error": error}
    return {"reaped": True, "reap_timed_out": False, "kill_error": error}


def run_bounded_subprocess(
    command, output, timeout, *, expected_config, log_limit=LOG_LIMIT_BYTES
):
    """Drain both pipes continuously with bounded retained logs and memory."""
    logs = {
        name: {"bytes": 0, "retained_bytes": 0, "truncated": False, "error": None}
        for name in ("stdout", "stderr")
    }
    began = time.monotonic()
    deadline = began + timeout
    # Cleanup uses part of this same budget, never a fresh unbounded wait.
    workload_deadline = deadline - min(1.0, timeout * 0.1)
    process = subprocess.Popen(
        command, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    cleanup_attempted = False
    cleanup = {"reaped": False, "reap_timed_out": False, "kill_error": None}
    try:

        def drain(name, pipe):
            record = logs[name]
            try:
                with pipe, (output / f"{name}.log").open("xb") as stream:
                    while chunk := pipe.read(65536):
                        record["bytes"] += len(chunk)
                        retained = chunk[: max(0, log_limit - record["retained_bytes"])]
                        stream.write(retained)
                        record["retained_bytes"] += len(retained)
                        record["truncated"] = record["bytes"] > log_limit
            except BaseException as error:
                record["error"] = repr(error)

        readers = [
            threading.Thread(target=drain, args=(name, pipe), daemon=True)
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr))
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            returncode = process.wait(
                timeout=max(0.0, workload_deadline - time.monotonic())
            )
            cleanup["reaped"] = True
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_attempted = True
            cleanup = terminate_and_reap(process, deadline)
            returncode = process.returncode
        for reader in readers:
            reader.join(timeout=max(0.0, deadline - time.monotonic()))
        logs_complete = all(not reader.is_alive() for reader in readers)
    finally:
        if process.poll() is None and not cleanup_attempted:
            cleanup = terminate_and_reap(process, deadline)
    # A descendant retaining a pipe must not keep the controller alive. Such
    # incomplete capture fails; daemon readers own/close their own pipe objects.
    logs = {name: record.copy() for name, record in logs.items()}
    elapsed = time.monotonic() - began
    completion = (
        validate_completion(output, expected_config, process.pid, elapsed)
        if cleanup["reaped"]
        else {"valid": False, "error": "child was not reaped", "sha256": None}
    )
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        **cleanup,
        "completion": completion,
        "elapsed_seconds": time.monotonic() - began,
        "log_limit_bytes": log_limit,
        "logs_complete": logs_complete,
        "logs": logs,
        "passed": returncode == 0
        and not timed_out
        and cleanup["reaped"]
        and cleanup["kill_error"] is None
        and completion["valid"]
        and logs_complete
        and all(
            not record["truncated"] and record["error"] is None
            for record in logs.values()
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_int(lower: int, upper: int):
    def parse(value: str) -> int:
        number = int(value)
        if not lower <= number <= upper:
            raise argparse.ArgumentTypeError(f"must be between {lower} and {upper}")
        return number

    return parse


def stable_report(report):
    """All public report fields, with exactly one explicitly volatile omission."""
    import struct
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = report.to_dict()
    # Live memory availability changes while independent matrices are resident.
    # Config, row order/identities, every metric, diagnostics, significance,
    # decisions, warnings and all remaining backend facts are compared.
    if result["backend"] is not None:
        result["backend"].pop("memory_free_mb", None)

    def encode(value):
        if isinstance(value, float):
            return {"binary64_bits": struct.pack(">d", value).hex()}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or isinstance(value, (str, int, bool)):
            return value
        raise TypeError(f"unhandled deterministic report field: {type(value)}")

    return json.dumps(encode(result), sort_keys=True, separators=(",", ":")).encode()


def child(config: dict) -> int:
    import ctypes
    import importlib
    import os

    if config_sha256(config) != config["config_sha256"]:
        raise RuntimeError("child configuration hash mismatch")
    if {name: sha256(ROOT / name) for name in SOURCE_FILES} != config["source_sha256"]:
        raise RuntimeError("runner sources changed before child startup")
    payload = Path(config["payload"])
    if (
        not config.get("acknowledge_fixed_candidate")
        or config["payload_sha256"] == KNOWN_UNFIXED_CUDA_RC2
    ):
        raise RuntimeError("only an explicitly acknowledged fixed candidate may run")
    if sha256(payload) != config["payload_sha256"]:
        raise RuntimeError("candidate payload changed before child startup")
    for key in ("legacy_shim", "numeric_shim"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise RuntimeError(f"{key} changed before child startup")
    os.environ[f"GAFIME_{config['backend'].upper()}_V1_LIB"] = str(payload)
    os.environ["GAFIME_V1_ANALYZE_CACHE_SIZE"] = "0"
    import numpy as np
    import gafime
    from gafime import ComputeBudget, CompileFlags, EngineConfig

    core = importlib.import_module("gafime.gafime_py")
    core_path = Path(core.__file__).resolve()
    identity = {
        "event": "identity",
        "pid": os.getpid(),
        "python": str(Path(sys.executable).resolve()),
        "core": str(core_path),
        "core_sha256": sha256(core_path),
        "config": config,
    }
    print(json.dumps(identity), flush=True)
    # Keep this exact loaded payload alive while consumer dlopen/dlclose cycles
    # run. Both shims receive this same canonical path as the Core selector.
    payload_pin = ctypes.CDLL(str(payload))
    functions = {}
    pins = [payload_pin]
    backend_kind = b"2" if config["backend"] == "cuda" else b"3"
    for abi, key in (("abi10", "legacy_shim"), ("abi11", "numeric_shim")):
        library = ctypes.CDLL(config[key])
        pins.append(library)
        version = "1_0" if abi == "abi10" else "1_1"
        function = getattr(library, f"gafime_abi_{version}_consumer_main")
        function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        function.restype = ctypes.c_int
        arguments = [b"consumer", str(payload).encode(), backend_kind]
        if abi == "abi11":
            arguments.append(b"3")
        argv = (ctypes.c_char_p * len(arguments))(*arguments)
        functions[abi] = (function, argv)

    rng = np.random.default_rng(0)
    dtype = np.float64 if config["precision"] == "fp64" else np.float32
    features = rng.normal(size=(4096, 8)).astype(dtype)
    target = rng.normal(size=4096).astype(dtype)
    engine_config = EngineConfig(
        backend=config["backend"],
        precision=config["precision"],
        permutation_tests=0,
        num_repeats=1,
        metric_names=("pearson", "r2"),
        budget=ComputeBudget(max_comb_size=2),
    )
    graph = config["case"].startswith("graph-")

    def primary():
        artifact = gafime.compile(
            features, target, config=engine_config, flags=CompileFlags(graph=graph)
        )
        try:
            report = artifact.analyze()
            if bool(artifact.graph_replayed) != graph:
                raise AssertionError("graph replay/fallback contract mismatch")
            if (
                report.backend is None
                or report.backend.selected_backend != config["backend"]
            ):
                raise AssertionError("unexpected backend fallback")
            return stable_report(report)
        finally:
            artifact.close()

    has_primary = config["case"] != "foreign-only"
    reference = primary() if has_primary else None
    selected = selected_abis(config["case"])
    stop = threading.Event()
    window = {}

    def begin_window():
        window["started_at"] = time.monotonic()
        window["deadline"] = window["started_at"] + config["seconds"]

    start = threading.Barrier(
        config["workers"] * len(selected) + 1, action=begin_window
    )
    results_lock = threading.Lock()
    progress = {
        f"{abi}-{worker}": CallProgress()
        for abi in selected
        for worker in range(config["workers"])
    }
    failures = []

    def foreign(abi, worker):
        function, argv = functions[abi]
        key = f"{abi}-{worker}"

        def invoke():
            # CDLL releases the GIL; the product's attachment policy is unchanged.
            status = function(len(argv), argv)
            if status != 0:
                with results_lock:
                    failures.append({"worker": key, "status": status})
                    stop.set()

        try:
            start.wait()
            run_until_deadline(invoke, window["deadline"], stop, progress[key])
        except BaseException as error:
            with results_lock:
                failures.append({"worker": key, "exception": repr(error)})
                stop.set()

    workers = [
        threading.Thread(target=foreign, args=(abi, worker))
        for abi in selected
        for worker in range(config["workers"])
    ]
    for worker in workers:
        worker.start()
    primary_progress = CallProgress()
    primary_error = None
    start.wait()

    def invoke_primary():
        if primary() != reference:
            raise AssertionError(
                "full deterministic report differs from serial reference"
            )

    try:
        if has_primary:
            run_until_deadline(
                invoke_primary,
                window["deadline"],
                stop,
                primary_progress,
                pause=lambda: time.sleep(0.001),
            )
        else:
            while time.monotonic() < window["deadline"] and not stop.is_set():
                time.sleep(0.001)
    except BaseException as error:
        primary_error = repr(error)
    finally:
        stop.set()
        # The controller enforces the hard deadline even if native work hangs.
        for worker in workers:
            worker.join()
    elapsed = time.monotonic() - window["started_at"]
    summaries = {
        key: progress_summary(value, window["started_at"], window["deadline"])
        for key, value in progress.items()
    }
    incomplete = [
        key for key, summary in summaries.items() if not summary["covered_interval"]
    ]
    primary_summary = (
        progress_summary(primary_progress, window["started_at"], window["deadline"])
        if has_primary
        else None
    )
    if primary_summary is not None and not primary_summary["covered_interval"]:
        incomplete.append("primary")
    passed = (
        not failures
        and primary_error is None
        and not incomplete
        and (not has_primary or primary_progress.calls > 0)
    )
    result = {
        "event": "result",
        "status": "pass" if passed else "fail",
        "requested_seconds": config["seconds"],
        "elapsed_seconds": elapsed,
        "primary_calls": primary_progress.calls,
        "primary_progress": primary_summary,
        "foreign_calls": {key: value.calls for key, value in progress.items()},
        "foreign_progress": summaries,
        "incomplete_interval_workers": incomplete,
        "foreign_failures": failures,
        "primary_error": primary_error,
        "parity_sha256": hashlib.sha256(reference).hexdigest() if reference else None,
        "excluded_report_fields": ["backend.memory_free_mb"],
    }
    write_completion(Path(config["output_dir"]), config, identity, result)
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="installed candidate Core interpreter",
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--legacy-shim", type=Path, required=True)
    parser.add_argument("--numeric-shim", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--precision", choices=("fp32", "mixed", "fp64"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seconds", type=bounded_int(1, 20), default=2)
    parser.add_argument(
        "--workers",
        type=bounded_int(1, 3),
        default=1,
        help="foreign workers per selected ABI generation",
    )
    parser.add_argument("--timeout", type=bounded_int(5, 120), default=45)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory; never overwrites logs",
    )
    parser.add_argument(
        "--acknowledge-fixed-candidate",
        action="store_true",
        help="operator confirms candidate build provenance and safe hardware preflight",
    )
    args = parser.parse_args()
    if not args.acknowledge_fixed_candidate:
        parser.error(
            "explicit fixed-candidate acknowledgement is required; no baseline stress is permitted"
        )
    if args.timeout <= args.seconds:
        parser.error("--timeout must exceed --seconds to allow setup and teardown")
    config = vars(args).copy()
    for key in ("python", "payload", "legacy_shim", "numeric_shim", "output_dir"):
        config[key] = str(config[key].resolve())
    for key in ("payload", "legacy_shim", "numeric_shim"):
        config[key + "_sha256"] = sha256(Path(config[key]))
    if config["payload_sha256"] != args.expected_payload_sha256.lower():
        parser.error("candidate payload SHA256 mismatch")
    if config["payload_sha256"] == KNOWN_UNFIXED_CUDA_RC2:
        parser.error("the known unfixed RC2 payload is not permitted")
    config["source_sha256"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    for key, command in (
        ("source_head", ["git", "rev-parse", "HEAD"]),
        ("source_status", ["git", "status", "--short"]),
    ):
        config[key] = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    config["run_id"] = secrets.token_hex(16)
    config["config_sha256"] = config_sha256(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "identity.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [
        config["python"],
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        json.dumps(config),
    ]
    status = run_bounded_subprocess(
        command, output, args.timeout, expected_config=config
    )
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"artifacts": str(output), **status}))
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        raise SystemExit(child(json.loads(sys.argv[2])))
    raise SystemExit(main())
