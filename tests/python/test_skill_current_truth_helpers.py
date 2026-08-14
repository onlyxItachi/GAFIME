from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"


def _load(relative: str, name: str):
    path = SKILLS / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurrentTruthHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.health = _load(
            "check-install/scripts/health_check.py", "skill_health_check"
        )
        cls.profiler = _load(
            "dataset-profiler/scripts/profile_dataset.py", "skill_dataset_profiler"
        )
        cls.interpreter = _load(
            "interpret-results/scripts/explain_report.py", "skill_interpreter"
        )
        cls.platform = _load(
            "platform-detect/scripts/platform_detect.py", "skill_platform_detect"
        )
        cls.troubleshooter = _load(
            "troubleshoot-backend/scripts/diagnose_backends.py",
            "skill_backend_diagnostics",
        )
        cls.time_series = _load(
            "time-series-setup/scripts/detect_time_structure.py",
            "skill_time_series_detector",
        )
        cls.benchmark = _load(
            "benchmark-vs-manual/scripts/compare_approaches.py",
            "skill_benchmark_manual",
        )
        cls.generator = _load(
            "build-pipeline/scripts/generate_pipeline.py", "skill_pipeline_generator"
        )
        cls.validator = _load(
            "validate-features/scripts/validate_features.py", "skill_feature_validator"
        )

    def test_skill_guidance_has_no_known_current_truth_regressions(self) -> None:
        guidance = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SKILLS.glob("*/SKILL.md"))
        )
        for stale in (
            "Decision-path permutation significance is unavailable",
            "cudart64_13.dll",
            "gafime-cuda" + "-rt",
            "gafime-rocm" + "-bundled",
            "gafime-metal",
        ):
            self.assertNotIn(stale, guidance)
        self.assertIn("per-permuted-target path rediscovery", guidance)
        self.assertIn("nvcudart_hybrid64.dll", guidance)

    def test_beta2_install_guidance_is_prospective_until_publication(self) -> None:
        for module in (self.health, self.platform, self.troubleshooter, self.benchmark):
            self.assertEqual(module.RELEASE_STATUS, "not_yet_published")

        platform_fields = self.platform._release_install_fields(
            self.platform.WHEN_PUBLISHED_CUDA_INSTALL
        )
        self.assertEqual(platform_fields["release_status"], "not_yet_published")
        self.assertIn("not yet published", platform_fields["current_install_guidance"])
        self.assertEqual(
            platform_fields["when_published_install"],
            'pip install "gafime==1.0.0b2" "gafime-cuda==1.0.0b2"',
        )
        self.assertNotIn("recommended_install", platform_fields)

        missing = self.benchmark._missing_gafime_install_record()
        self.assertIn("not yet published", missing["error"])
        self.assertEqual(missing["release_status"], "not_yet_published")
        self.assertEqual(
            missing["when_published_install"],
            "pip install 'gafime[sklearn]==1.0.0b2'",
        )

        for relative in (
            "build-pipeline/SKILL.md",
            "check-install/SKILL.md",
            "platform-detect/SKILL.md",
            "troubleshoot-backend/SKILL.md",
        ):
            guidance = (SKILLS / relative).read_text(encoding="utf-8").lower()
            self.assertIn("not yet published", guidance)
            self.assertIn("once beta.2 is published", guidance)

    def test_payload_versions_must_match_core_exactly(self) -> None:
        installed = {
            "gafime": "1.0.0b2",
            "gafime-cuda": "1.0.0b2",
            "gafime-rocm": None,
        }
        self.assertIn(
            '"gafime-cuda": "1.0.0b2"',
            self.health._validate_distribution_versions(installed),
        )
        installed["gafime-cuda"] = "1.0.0b1"
        with self.assertRaisesRegex(RuntimeError, "exact-version mismatch"):
            self.health._validate_distribution_versions(installed)
        with self.assertRaisesRegex(RuntimeError, "runtime metadata mismatch"):
            self.health._validate_distribution_versions(
                {"gafime": "1.0.0b2"}, runtime_version="1.0.0b1"
            )

    def test_health_check_fails_closed_outside_release_python_matrix(self) -> None:
        self.assertIn("3.10-3.14", self.health._validate_python_version((3, 10, 0)))
        self.assertIn("3.10-3.14", self.health._validate_python_version((3, 14, 9)))
        for unsupported in ((3, 9, 20), (3, 15, 0), (4, 0, 0)):
            with self.assertRaisesRegex(RuntimeError, "3.10 through 3.14"):
                self.health._validate_python_version(unsupported)
        with self.assertRaisesRegex(RuntimeError, "supports CPython only"):
            self.health._validate_python_version((3, 10, 0), implementation="PyPy")

    def test_profiler_precision_and_small_batch_are_conservative(self) -> None:
        self.assertEqual(self.profiler._resident_element_bytes("mixed"), 4)
        self.assertEqual(self.profiler._resident_element_bytes("fp64"), 8)
        self.assertEqual(
            self.profiler._recommended_batch_size(
                n_rows=10_000, bytes_per_row=1024, usable_vram_bytes=100 * 1024
            ),
            100,
        )

    def test_dataset_tools_honor_precision_and_explicit_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gafime-skill-test-") as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            self.profiler.pl.DataFrame(
                {
                    "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
                    "feature": [1.0, 2.0, 3.0],
                    "outcome": [0.0, 1.0, 0.0],
                }
            ).write_csv(path)
            profile = self.profiler.profile_dataset(
                str(path), target_col="outcome", precision="fp64"
            )
            self.assertEqual(profile["memory"]["resident_element_bytes"], 8)
            structure = self.time_series.detect_time_structure(
                str(path), target_col="outcome"
            )
            self.assertEqual(structure["target_used_for_estimate"], "outcome")
            self.assertEqual(
                structure["feature_estimates"]["numeric_input_features"], 1
            )

    def test_interpreter_reports_precision_and_current_decision_path_support(
        self,
    ) -> None:
        explained = self.interpreter.explain_report(
            {
                "config": {"precision": "mixed"},
                "backend": {
                    "selected_backend": "core",
                    "effective_precision": "mixed",
                    "storage_dtype": "float32",
                    "reduction_dtype": "float64",
                    "result_dtype": "float64",
                },
            }
        )
        self.assertEqual(explained["overview"]["effective_precision"], "mixed")
        self.assertTrue(
            any(
                "rediscovery for every permuted target" in limit
                for limit in explained["interpretation_limits"]
            )
        )

    def test_platform_and_backend_diagnostics_flag_version_skew(self) -> None:
        payloads = {"gafime": "1.0.0b2", "gafime-cuda": "1.0.0b1"}
        self.assertEqual(len(self.platform._payload_version_warnings(payloads)), 1)
        self.assertEqual(
            len(self.troubleshooter._payload_version_warnings(payloads)), 1
        )
        self.assertEqual(
            self.troubleshooter._diagnostic_precision("metal", "mixed"), "fp32"
        )
        self.assertEqual(
            self.troubleshooter._diagnostic_precision("cuda", "mixed"), "mixed"
        )

    def test_manual_divide_matches_selector_denominator_rule(self) -> None:
        left = np.asarray([2.0, 2.0], dtype=np.float32)
        right = np.asarray([-2.0, -1e-12], dtype=np.float32)
        values = self.benchmark._divide_values(left, right)
        self.assertEqual(float(values[0]), -1.0)
        self.assertGreater(float(values[1]), 0.0)

    def test_generated_pipeline_carries_precision_and_compiles(self) -> None:
        script = self.generator.generate_pipeline_script(
            "classification", precision="fp64"
        )
        compile(script, "generated-pipeline.py", "exec")
        self.assertIn('precision="fp64"', script)
        self.assertIn(
            'backend_capabilities("auto", probe=True, precision="fp64")', script
        )
        self.assertIn("astype(np.float64)", script)
        self.assertNotIn("astype(np.float32)", script)

    def test_validation_uses_neutral_status_and_paired_finite_rows(self) -> None:
        x = np.arange(60, dtype=np.float32).reshape(20, 3)
        x[0, 0] = np.nan
        y = np.arange(20, dtype=np.float32)
        report = self.validator.validate_interactions(
            x,
            y,
            [(0, 1)],
            n_random_baselines=5,
            precision="mixed",
        )
        self.assertIn(
            report["interactions"][0]["status"],
            {"HEURISTIC_PASS", "HEURISTIC_INCONCLUSIVE"},
        )
        self.assertNotIn("genuine_count", report)
        row = report["interactions"][0]
        self.assertEqual(row["finite_train_rows"] + row["finite_holdout_rows"], 19)


if __name__ == "__main__":
    unittest.main()
