"""Host-only regression for complete ordinary-export coverage; no GPU imports."""

from pathlib import Path
from copy import deepcopy
import re
import unittest

from execution_coordination_regression import stable_report


ROOT = Path(__file__).resolve().parents[2]
EXPORT = re.compile(
    r"GAFIME_GPU_API\s+(int|void)\s+(gafime_gpu_\w+)\s*\([^;]*?\)\s*(?:try\s*)?\{",
    re.DOTALL,
)
GUARD = "gafime_gpu::PayloadExecutionGuard execution_guard(payload_execution_mutex);"
COMMON = {
    "device_info", "graph_capability", "matrix_alloc", "matrix_upload",
    "matrix_update_target", "matrix_free", "interaction_diagnostics",
    "execution_memory_peak", "execute", "numeric_routes_v2", "matrix_alloc_v2",
    "matrix_upload_v2", "matrix_update_target_v2", "execute_v2",
    "execution_memory_peak_v2", "permutation_memory_peak_v2",
    "permutation_pvalues_v2", "interaction_diagnostics_v2", "matrix_free_v2",
}


class ExecutionGateSourceTest(unittest.TestCase):
    def test_every_outer_export_enters_once_before_other_work(self):
        for backend, filename in (("cuda", "precision_launcher.cu"), ("rocm", "launcher.hip")):
            with self.subTest(backend=backend):
                source = (ROOT / "src" / backend / filename).read_text()
                exports = list(EXPORT.finditer(source))
                expected = COMMON | ({"permutation_memory_peak", "permutation_pvalues"}
                                     if backend == "cuda" else set())
                self.assertEqual({match[2].removeprefix("gafime_gpu_") for match in exports}, expected)
                self.assertEqual(source.count(GUARD), len(exports))
                self.assertEqual(source.count("std::mutex payload_execution_mutex;"), 1)
                for match in exports:
                    # First statements are the guard and its fail-closed exit.
                    # Consequently ScopedDevice and all other locals die first.
                    failure = "return;" if match[1] == "void" else "return GAFIME_STATUS_DEVICE_ERROR;"
                    prefix = source[match.end():].lstrip()
                    self.assertTrue(prefix.startswith(GUARD), match[2])
                    self.assertTrue(prefix[len(GUARD):].lstrip().startswith(
                        f"if (!execution_guard.acquired()) {failure}"), match[2])
                # Adapters must use shared internals, never another locked export.
                # Catch new runtime-touching exports even if the allowlist was not
                # updated, and reject any direct export-to-export call.
                all_references = re.findall(r"\b(gafime_gpu_\w+)\s*\(", source)
                self.assertCountEqual(all_references, [match[2] for match in exports])

    def test_header_is_in_payload_source_distribution(self):
        stage = (ROOT / ".github/scripts/stage_gpu_payload.py").read_text()
        composition = (ROOT / "tests/release_measure/artifact_01_release_composition.py").read_text()
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
            "interactions": [{"candidate_id": "0", "metrics": {"pearson": 0.0},
                              "interaction_overflow_rows": 0}],
            "warnings": [], "decision": {"signal_detected": True},
        }
        reference = stable_report(Report(fields))
        modified = deepcopy(fields)
        modified["backend"]["memory_free_mb"] = 50
        self.assertEqual(reference, stable_report(Report(modified)))
        for key, value in (("candidate_id", "1"), ("interaction_overflow_rows", 1),
                           ("metrics", {"pearson": -0.0})):
            modified = deepcopy(fields)
            modified["interactions"][0][key] = value
            self.assertNotEqual(reference, stable_report(Report(modified)))
        modified = deepcopy(fields)
        modified["new_future_report_field"] = "must also compare"
        self.assertNotEqual(reference, stable_report(Report(modified)))
        self.assertEqual(fields["backend"]["memory_free_mb"], 100)


if __name__ == "__main__":
    unittest.main()
