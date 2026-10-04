"""Opt-in installed-package/native-ABI collision regression. Never a default test.

Run --help without GPU imports. The controller launches one bounded isolated
Python subprocess; only that subprocess imports GAFIME or opens the payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


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
    import threading
    import time

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
    print(
        json.dumps(
            {
                "event": "identity",
                "core": str(core_path),
                "core_sha256": sha256(core_path),
                "config": config,
            }
        ),
        flush=True,
    )
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
    selected = (
        ["abi10"]
        if config["case"] == "graph-abi10"
        else ["abi11"]
        if config["case"] == "graph-abi11"
        else ["abi10", "abi11"]
    )
    stop = threading.Event()
    start = threading.Barrier(config["workers"] * len(selected) + 1)
    results_lock = threading.Lock()
    counts = {
        f"{abi}-{worker}": 0 for abi in selected for worker in range(config["workers"])
    }
    failures = []

    def foreign(abi, worker):
        function, argv = functions[abi]
        key = f"{abi}-{worker}"
        try:
            start.wait()
            # At least one call per worker, at most 512. CDLL releases the GIL
            # for these calls; no Python detachment feature is involved.
            for _ in range(512):
                status = function(len(argv), argv)
                with results_lock:
                    counts[key] += 1
                    if status != 0:
                        failures.append({"worker": key, "status": status})
                        stop.set()
                if stop.is_set():
                    return
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
    primary_count = 0
    primary_error = None
    start.wait()
    deadline = time.monotonic() + config["seconds"]
    try:
        while time.monotonic() < deadline and not stop.is_set():
            if has_primary:
                if primary() != reference:
                    raise AssertionError(
                        "full deterministic report differs from serial reference"
                    )
                primary_count += 1
            # Let independent callers queue their next native entries.
            time.sleep(0.001)
    except BaseException as error:
        primary_error = repr(error)
    finally:
        stop.set()
        # The controller enforces the hard deadline even if native work hangs.
        for worker in workers:
            worker.join()
    passed = (
        not failures
        and primary_error is None
        and all(counts.values())
        and (not has_primary or primary_count > 0)
    )
    print(
        json.dumps(
            {
                "event": "result",
                "status": "pass" if passed else "fail",
                "primary_calls": primary_count,
                "foreign_calls": counts,
                "foreign_failures": failures,
                "primary_error": primary_error,
                "parity_sha256": hashlib.sha256(reference).hexdigest()
                if reference
                else None,
                "excluded_report_fields": ["backend.memory_free_mb"],
            }
        ),
        flush=True,
    )
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
    parser.add_argument("--seconds", type=bounded_int(1, 10), default=2)
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
    try:
        result = subprocess.run(
            command,
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        stdout, stderr = result.stdout, result.stderr
        status = {"returncode": result.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        status = {"returncode": None, "timed_out": True}
    (output / "stdout.log").write_bytes(stdout)
    (output / "stderr.log").write_bytes(stderr)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"artifacts": str(output), **status}))
    return 0 if status == {"returncode": 0, "timed_out": False} else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        raise SystemExit(child(json.loads(sys.argv[2])))
    raise SystemExit(main())
