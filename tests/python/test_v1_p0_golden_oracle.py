from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "tests" / "golden" / "generate_golden.py"
GOLDEN_DIR = ROOT / "tests" / "golden"


def _golden_module():
    spec = importlib.util.spec_from_file_location("gafime_v1_p0_golden", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v1_p0_golden_oracle_matches_legacy_core_outputs():
    golden = _golden_module()
    try:
        actual_cases = golden.build_golden_cases()
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    assert set(actual_cases) == {"continuous_core", "time_series_core"}
    for name, actual in actual_cases.items():
        expected = golden.load_golden_case(GOLDEN_DIR / f"{name}.json")
        golden.compare_golden(actual, expected, name)
