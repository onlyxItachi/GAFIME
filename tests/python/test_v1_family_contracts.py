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
    assert not families["decision_path"].supported
    assert not families["time_series"].supported
    assert all(not family.python_candidate_loop for family in families.values())


def test_unsupported_families_raise_explicit_v1_errors():
    with pytest.raises(V1UnsupportedError, match="no native device kernel"):
        gafime.require_family_supported("decision_path")
    with pytest.raises(V1UnsupportedError, match="no native device kernel"):
        gafime.require_family_supported("time_series")
