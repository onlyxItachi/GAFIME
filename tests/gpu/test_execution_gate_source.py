"""Host-only regression for complete ordinary-export coverage; no GPU imports."""

import argparse
import json
from pathlib import Path
from copy import deepcopy
import re
import subprocess
import sys
import tempfile
import threading
import unittest

from execution_coordination_regression import (
    CallProgress,
    COMPLETION_FILE,
    COMPLETION_LIMIT_BYTES,
    bounded_int,
    config_sha256,
    progress_summary,
    run_bounded_subprocess,
    run_until_deadline,
    selected_abis,
    sha256,
    stable_report,
    terminate_and_reap,
    validate_completion,
    write_completion,
)


ROOT = Path(__file__).resolve().parents[2]
EXPORT = re.compile(
    r"GAFIME_GPU_API\s+(int|void)\s+(gafime_gpu_\w+)\s*\([^;]*?\)\s*(?:try\s*)?\{",
    re.DOTALL,
)
GUARD = "gafime_gpu::PayloadExecutionGuard execution_guard(payload_execution_mutex);"
COMMON = {
    "device_info",
    "graph_capability",
    "matrix_alloc",
    "matrix_upload",
    "matrix_update_target",
    "matrix_free",
    "interaction_diagnostics",
    "execution_memory_peak",
    "execute",
    "numeric_routes_v2",
    "matrix_alloc_v2",
    "matrix_upload_v2",
    "matrix_update_target_v2",
    "execute_v2",
    "execution_memory_peak_v2",
    "permutation_memory_peak_v2",
    "permutation_pvalues_v2",
    "interaction_diagnostics_v2",
    "matrix_free_v2",
}


def synthetic_completion(case="graph-both"):
    """Host protocol fixture, explicitly not native execution evidence."""
    config = {
        "run_id": "host-only-fixture",
        "python": str(Path(sys.executable).resolve()),
        "case": case,
        "seconds": 1,
        "timeout": 5,
        "workers": 1,
        "payload_sha256": "a" * 64,
        "source_sha256": {"synthetic-source": "b" * 64},
    }
    config["config_sha256"] = config_sha256(config)
    progress = progress_summary(CallProgress(10, 0.0, 1.0, 0.8), 0.0, 1.0)
    workers = {f"{abi}-0": deepcopy(progress) for abi in selected_abis(case)}
    has_primary = case != "foreign-only"
    identity = {
        "event": "identity",
        "pid": 123,
        "python": config["python"],
        "core": str(Path(__file__).resolve()),
        "core_sha256": sha256(Path(__file__)),
        "config": deepcopy(config),
    }
    result = {
        "event": "result",
        "status": "pass",
        "requested_seconds": 1,
        "elapsed_seconds": 1.01,
        "primary_calls": 10 if has_primary else 0,
        "primary_progress": deepcopy(progress) if has_primary else None,
        "foreign_calls": {key: 10 for key in workers},
        "foreign_progress": workers,
        "incomplete_interval_workers": [],
        "foreign_failures": [],
        "primary_error": None,
        "parity_sha256": "c" * 64 if has_primary else None,
        "excluded_report_fields": ["backend.memory_free_mb"],
    }
    return config, {
        "version": 1,
        "run_id": config["run_id"],
        "config_sha256": config["config_sha256"],
        "identity": identity,
        "result": result,
    }


class ExecutionGateSourceTest(unittest.TestCase):
    def test_every_outer_export_enters_once_before_other_work(self):
        for backend, filename in (
            ("cuda", "precision_launcher.cu"),
            ("rocm", "launcher.hip"),
        ):
            with self.subTest(backend=backend):
                source = (ROOT / "src" / backend / filename).read_text()
                exports = list(EXPORT.finditer(source))
                expected = COMMON | (
                    {"permutation_memory_peak", "permutation_pvalues"}
                    if backend == "cuda"
                    else set()
                )
                self.assertEqual(
                    {match[2].removeprefix("gafime_gpu_") for match in exports},
                    expected,
                )
                self.assertEqual(source.count(GUARD), len(exports))
                self.assertEqual(source.count("std::mutex payload_execution_mutex;"), 1)
                for match in exports:
                    # First statements are the guard and its fail-closed exit.
                    # Consequently ScopedDevice and all other locals die first.
                    failure = (
                        "return;"
                        if match[1] == "void"
                        else "return GAFIME_STATUS_DEVICE_ERROR;"
                    )
                    prefix = source[match.end() :].lstrip()
                    self.assertTrue(prefix.startswith(GUARD), match[2])
                    self.assertTrue(
                        prefix[len(GUARD) :]
                        .lstrip()
                        .startswith(f"if (!execution_guard.acquired()) {failure}"),
                        match[2],
                    )
                # Adapters must use shared internals, never another locked export.
                # Catch new runtime-touching exports even if the allowlist was not
                # updated, and reject any direct export-to-export call.
                all_references = re.findall(r"\b(gafime_gpu_\w+)\s*\(", source)
                self.assertCountEqual(all_references, [match[2] for match in exports])

    def test_header_is_in_payload_source_distribution(self):
        stage = (ROOT / ".github/scripts/stage_gpu_payload.py").read_text()
        composition = (
            ROOT / "tests/release_measure/artifact_01_release_composition.py"
        ).read_text()
        self.assertIn('"gpu_execution_gate.hpp"', stage)
        self.assertEqual(composition.count('"src/common/gpu_execution_gate.hpp"'), 2)

    def test_report_comparison_preserves_all_fields_except_live_free_memory(self):
        class Report:
            def __init__(self, data):
                self.data = data

            def to_dict(self):
                return deepcopy(self.data)

        fields = {
            "backend": {"memory_free_mb": 100, "effective_precision": "mixed"},
            "interactions": [
                {
                    "candidate_id": "0",
                    "metrics": {"pearson": 0.0},
                    "interaction_overflow_rows": 0,
                }
            ],
            "warnings": [],
            "decision": {"signal_detected": True},
        }
        reference = stable_report(Report(fields))
        modified = deepcopy(fields)
        modified["backend"]["memory_free_mb"] = 50
        self.assertEqual(reference, stable_report(Report(modified)))
        for key, value in (
            ("candidate_id", "1"),
            ("interaction_overflow_rows", 1),
            ("metrics", {"pearson": -0.0}),
        ):
            modified = deepcopy(fields)
            modified["interactions"][0][key] = value
            self.assertNotEqual(reference, stable_report(Report(modified)))
        modified = deepcopy(fields)
        modified["new_future_report_field"] = "must also compare"
        self.assertNotEqual(reference, stable_report(Report(modified)))
        self.assertEqual(fields["backend"]["memory_free_mb"], 100)


class TimedRunnerTest(unittest.TestCase):
    def test_foreign_loop_runs_past_512_calls_until_deadline(self):
        now = [0.0]
        progress = CallProgress()

        def operation():
            now[0] += 0.001

        run_until_deadline(
            operation, 1.0, threading.Event(), progress, clock=lambda: now[0]
        )
        self.assertGreater(progress.calls, 512)
        self.assertGreaterEqual(now[0], 1.0)
        self.assertTrue(progress_summary(progress, 0.0, 1.0)["covered_interval"])

    def test_stop_and_expired_deadline_do_not_start_more_calls(self):
        now = [0.0]
        stop = threading.Event()
        progress = CallProgress()

        def operation():
            now[0] += 0.01
            if now[0] >= 0.02:
                stop.set()

        run_until_deadline(operation, 1.0, stop, progress, clock=lambda: now[0])
        self.assertEqual(progress.calls, 2)
        self.assertFalse(progress_summary(progress, 0.0, 1.0)["covered_interval"])
        expired = CallProgress()
        run_until_deadline(
            lambda: self.fail("started after deadline"),
            1.0,
            threading.Event(),
            expired,
            clock=lambda: 1.0,
        )
        self.assertEqual(expired.calls, 0)

    def test_both_abis_must_span_the_interval(self):
        summaries = {
            "abi10-0": progress_summary(CallProgress(512, 0.0, 0.2, 0.2), 0.0, 15.0),
            "abi11-0": progress_summary(CallProgress(100, 0.0, 15.1, 15.1), 0.0, 15.0),
        }
        self.assertEqual(
            [key for key, value in summaries.items() if not value["covered_interval"]],
            ["abi10-0"],
        )
        self.assertFalse(
            progress_summary(CallProgress(100, 10.0, 15.1, 5.1), 0.0, 15.0)[
                "covered_interval"
            ]
        )
        self.assertFalse(
            progress_summary(CallProgress(1, 0.0, 15.1, 15.1), 0.0, 15.0)[
                "covered_interval"
            ]
        )

    def test_log_capture_is_bounded_and_truncation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = run_bounded_subprocess(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; sys.stdout.write('x' * 4096); sys.stderr.write('y' * 4096)",
                ],
                output,
                5,
                expected_config=synthetic_completion()[0],
                log_limit=128,
            )
            self.assertEqual(status["returncode"], 0)
            self.assertTrue(status["logs_complete"])
            self.assertFalse(status["passed"])
            for name in ("stdout", "stderr"):
                self.assertEqual(status["logs"][name]["bytes"], 4096)
                self.assertTrue(status["logs"][name]["truncated"])
                self.assertEqual((output / f"{name}.log").stat().st_size, 128)

    def test_clean_exit_without_completion_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = run_bounded_subprocess(
                [sys.executable, "-I", "-c", "print('host-only')"],
                output,
                5,
                expected_config=synthetic_completion()[0],
            )
            self.assertEqual(status["returncode"], 0)
            self.assertFalse(status["passed"])
            self.assertFalse(status["completion"]["valid"])
            self.assertTrue(status["logs_complete"])
            self.assertEqual((output / "stdout.log").read_bytes(), b"host-only\n")

    def test_empty_clean_exit_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            status = run_bounded_subprocess(
                [sys.executable, "-I", "-c", "pass"],
                Path(directory),
                5,
                expected_config=synthetic_completion()[0],
            )
            self.assertEqual(status["returncode"], 0)
            self.assertTrue(status["logs_complete"])
            self.assertFalse(status["passed"])
            self.assertIn("missing", status["completion"]["error"])

    def test_bounded_capture_accepts_valid_synthetic_child_protocol(self):
        config, record = synthetic_completion()
        script = (
            "import json, os, sys, time; from pathlib import Path; "
            "record = json.loads(sys.argv[1]); record['identity']['pid'] = os.getpid(); "
            "time.sleep(1.02); "
            f"Path({COMPLETION_FILE!r}).write_text(json.dumps(record)); "
            "print('host-only synthetic completion')"
        )
        with tempfile.TemporaryDirectory() as directory:
            status = run_bounded_subprocess(
                [sys.executable, "-I", "-c", script, json.dumps(record)],
                Path(directory),
                5,
                expected_config=config,
            )
            self.assertTrue(status["passed"], status)
            self.assertTrue(status["completion"]["valid"])
            self.assertTrue(status["reaped"])

    def test_controller_rejects_wrong_or_malformed_record_after_clean_exit(self):
        config, record = synthetic_completion()
        record["run_id"] = "another-run"
        for contents in ("{}", json.dumps(record)):
            with self.subTest(contents=contents[:40]):
                with tempfile.TemporaryDirectory() as directory:
                    status = run_bounded_subprocess(
                        [
                            sys.executable,
                            "-I",
                            "-c",
                            "import sys; from pathlib import Path; "
                            f"Path({COMPLETION_FILE!r}).write_text(sys.argv[1])",
                            contents,
                        ],
                        Path(directory),
                        5,
                        expected_config=config,
                    )
                    self.assertEqual(status["returncode"], 0)
                    self.assertTrue(status["logs_complete"])
                    self.assertFalse(status["completion"]["valid"])
                    self.assertFalse(status["passed"])

    def test_hard_timeout_fails_without_gpu_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            status = run_bounded_subprocess(
                [sys.executable, "-I", "-c", "import time; time.sleep(2)"],
                Path(directory),
                0.2,
                expected_config=synthetic_completion()[0],
            )
            self.assertTrue(status["timed_out"])
            self.assertFalse(status["passed"])
            self.assertTrue(status["reaped"])
            self.assertFalse(status["reap_timed_out"])
            self.assertLess(status["elapsed_seconds"], 1.0)

    def test_post_kill_reap_uses_only_remaining_budget(self):
        class NeverReaped:
            killed = False
            waits = []

            def kill(self):
                self.killed = True

            def wait(self, *, timeout):
                self.waits.append(timeout)
                raise subprocess.TimeoutExpired("host-only synthetic process", timeout)

        process = NeverReaped()
        status = terminate_and_reap(process, 10.0, clock=lambda: 9.9)
        self.assertTrue(process.killed)
        self.assertEqual(len(process.waits), 1)
        self.assertAlmostEqual(process.waits[0], 0.1)
        self.assertFalse(status["reaped"])
        self.assertTrue(status["reap_timed_out"])
        terminate_and_reap(process, 10.0, clock=lambda: 10.1)
        self.assertEqual(process.waits[-1], 0.0)

    def test_interval_accepts_twenty_seconds_but_remains_bounded(self):
        self.assertEqual(bounded_int(1, 20)("15"), 15)
        self.assertEqual(bounded_int(1, 20)("20"), 20)
        for value in ("0", "21"):
            with self.assertRaises(argparse.ArgumentTypeError):
                bounded_int(1, 20)(value)


class CompletionEvidenceTest(unittest.TestCase):
    def check_record(self, record, config=None, *, elapsed=2.0):
        if config is None:
            config = synthetic_completion()[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data = record if isinstance(record, (str, bytes)) else json.dumps(record)
            (output / COMPLETION_FILE).write_bytes(
                data.encode() if isinstance(data, str) else data
            )
            return validate_completion(output, config, 123, elapsed)

    def test_accepts_each_exact_synthetic_case(self):
        for case in (
            "graph-abi10",
            "graph-abi11",
            "graph-both",
            "eager-both",
            "foreign-only",
        ):
            with self.subTest(case=case):
                config, record = synthetic_completion(case)
                self.assertTrue(self.check_record(record, config)["valid"])

    def test_rejects_wrong_identity_and_configuration_hashes(self):
        changes = (
            (("run_id",), "another-run"),
            (("config_sha256",), "d" * 64),
            (("identity", "pid"), 124),
            (("identity", "python"), "/not/the/selected/interpreter"),
            (("identity", "core_sha256"), "d" * 64),
            (("identity", "config", "payload_sha256"), "d" * 64),
            (("identity", "config", "source_sha256"), {}),
            (("identity", "config", "case"), "foreign-only"),
            (("identity", "config", "workers"), 1.0),
        )
        for path, value in changes:
            with self.subTest(path=path):
                _, record = synthetic_completion()
                target = record
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(self.check_record(record)["valid"])

    def test_rejects_missing_malformed_and_duplicate_evidence(self):
        _, record = synthetic_completion()
        encoded = json.dumps(record)
        examples = (
            b"",
            b"not JSON",
            {},
            [],
            {key: value for key, value in record.items() if key != "identity"},
            {key: value for key, value in record.items() if key != "result"},
            encoded + encoded,
            encoded[:-1] + ', "result": ' + json.dumps(record["result"]) + "}",
            encoded.replace('"status": "pass"', '"status": "fail", "status": "pass"'),
            b" " * (COMPLETION_LIMIT_BYTES + 1),
        )
        for index, value in enumerate(examples):
            with self.subTest(index=index):
                self.assertFalse(self.check_record(value)["valid"])

    def test_recomputes_coverage_and_requires_full_parity(self):
        changes = (
            (("status",), "fail"),
            (("foreign_calls", "abi10-0"), 0),
            (("foreign_progress", "abi10-0", "last_call_offset_seconds"), 0.2),
            (("foreign_progress", "abi11-0", "first_call_offset_seconds"), 0.7),
            (("foreign_progress", "abi11-0", "active_span_seconds"), float("nan")),
            (("primary_progress", "covered_interval"), False),
            (("parity_sha256",), None),
            (("excluded_report_fields",), []),
            (("primary_error",), "failed"),
            (("foreign_failures",), [{"worker": "abi10-0", "status": 1}]),
            (("foreign_progress",), {}),
        )
        for path, value in changes:
            with self.subTest(path=path):
                _, record = synthetic_completion()
                target = record["result"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertFalse(self.check_record(record)["valid"])
        _, record = synthetic_completion()
        self.assertFalse(self.check_record(record, elapsed=0.1)["valid"])

    def test_writer_is_bounded_and_create_once(self):
        config, record = synthetic_completion()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_completion(output, config, record["identity"], record["result"])
            with self.assertRaises(FileExistsError):
                write_completion(output, config, record["identity"], record["result"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "size limit"):
                write_completion(
                    Path(directory),
                    config,
                    record["identity"],
                    "x" * COMPLETION_LIMIT_BYTES,
                )
            self.assertFalse((Path(directory) / COMPLETION_FILE).exists())


if __name__ == "__main__":
    unittest.main()
