"""Keep the bounded lifecycle diagnostic honest about installation and delivery."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _helper():
    path = (
        Path(__file__).resolve().parents[1]
        / "release_measure"
        / "semantic_02_discovery_sanity.py"
    )
    spec = importlib.util.spec_from_file_location("discovery_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_rejects_checkout_and_missing_module_paths(tmp_path):
    helper = _helper()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    module_file = checkout / "package.py"
    module_file.touch()
    with pytest.raises(AssertionError, match="external installation"):
        helper._assert_installed(SimpleNamespace(__file__=str(module_file)), checkout)
    with pytest.raises(AssertionError, match="external installation"):
        helper._assert_installed(SimpleNamespace(), checkout)
    installed = tmp_path / "installed.py"
    installed.touch()
    assert (
        helper._assert_installed(SimpleNamespace(__file__=str(installed)), checkout)
        == installed
    )


@pytest.mark.parametrize("precision", ["fp32", "mixed", "fp64"])
def test_diagnostic_checks_resident_evidence_and_unlabeled_arrow_delivery(precision):
    result = _helper().sample_run("core", precision, 1)
    assert result["backend"] == "core"
    assert result["precision"] == precision
    assert len(result["samples"]) == 2
    assert len(result["inference_values_sha256"]) == 64
