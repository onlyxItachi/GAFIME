from __future__ import annotations

from pathlib import Path
import sys

import pytest

_PYTHON_SRC = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SRC))

import gafime
from gafime.errors import V1UnsupportedError


def test_family_capabilities_are_declarative_without_python_loops():
    families = {family.name: family for family in gafime.available_families()}

    assert set(families) == {"continuous", "decision_path", "time_series"}
    assert families["continuous"].supported
    assert families["continuous"].cpu_kernel
    assert families["continuous"].cuda_kernel
    assert families["continuous"].rocm_kernel
    # decision_path + time_series are wired via native expansion + continuous
    # mining, so both are supported on CPU, CUDA, and ROCm.
    assert families["decision_path"].supported
    assert families["decision_path"].cpu_kernel
    assert families["decision_path"].cuda_kernel
    assert families["decision_path"].rocm_kernel
    assert families["time_series"].supported
    assert families["time_series"].cpu_kernel
    assert families["time_series"].cuda_kernel
    assert families["time_series"].rocm_kernel
    assert all(not family.python_candidate_loop for family in families.values())


def test_require_family_supported_accepts_wired_families_and_rejects_unknown():
    # decision_path + time_series are wired (expansion + continuous mining), so
    # require_family_supported returns their capability without raising.
    assert gafime.require_family_supported("decision_path").supported
    assert gafime.require_family_supported("time_series").supported
    # An unknown family name still raises an explicit v1 error.
    with pytest.raises(V1UnsupportedError):
        gafime.require_family_supported("nonexistent_family")
