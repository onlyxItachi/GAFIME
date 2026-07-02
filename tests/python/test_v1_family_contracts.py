from __future__ import annotations

import pytest

import gafime
from gafime.errors import V1UnsupportedError


def test_family_capabilities_are_declarative_without_python_loops():
    families = {family.name: family for family in gafime.available_families()}

    assert set(families) == {"continuous", "decision_path", "time_series"}
    assert families["continuous"].supported
    assert families["continuous"].cpu_kernel
    assert families["continuous"].cuda_kernel
    # decision_path + time_series are wired via native expansion + continuous
    # mining, so both are supported on CPU and CUDA.
    assert families["decision_path"].supported
    assert families["decision_path"].cpu_kernel and families["decision_path"].cuda_kernel
    assert families["time_series"].supported
    assert families["time_series"].cpu_kernel and families["time_series"].cuda_kernel
    assert all(not family.python_candidate_loop for family in families.values())


def test_require_family_supported_accepts_wired_families_and_rejects_unknown():
    # decision_path + time_series are wired (expansion + continuous mining), so
    # require_family_supported returns their capability without raising.
    assert gafime.require_family_supported("decision_path").supported
    assert gafime.require_family_supported("time_series").supported
    # An unknown family name still raises an explicit v1 error.
    with pytest.raises(V1UnsupportedError):
        gafime.require_family_supported("nonexistent_family")
