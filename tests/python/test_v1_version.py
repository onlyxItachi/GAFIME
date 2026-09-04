from __future__ import annotations

import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(PYTHON_SRC))

import gafime  # noqa: E402


def test_public_python_and_native_versions_match_pre_release_metadata():
    import gafime.gafime_py as native

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")

    assert gafime.__version__ == "1.0.0rc1"
    assert native.__version__ == gafime.__version__
    assert native.native_version() == gafime.__version__
    assert re.search(r'^version = "1\.0\.0rc1"$', pyproject, re.MULTILINE)
    assert re.search(r'^version = "1\.0\.0-rc\.1"$', cargo, re.MULTILINE)


def test_v1_polars_dependency_stays_on_the_supported_major():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    contract = (ROOT / "docs" / "contract.md").read_text(encoding="utf-8")

    assert re.search(r'^\s*"polars>=1\.3,<2",$', pyproject, re.MULTILINE)
    assert "GAFIME v1 deliberately supports `polars>=1.3,<2`" in contract
    assert "v1.1 or v1.2" in contract
