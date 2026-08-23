from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if (
    os.environ.get("GAFIME_TEST_INSTALLED_PACKAGE") != "1"
    and str(PYTHON_SRC) not in sys.path
):
    sys.path.insert(0, str(PYTHON_SRC))

import gafime  # noqa: E402


GENERATOR_PATH = ROOT / "docs" / "notebooks" / "generate_v1_api_reference.py"
NOTEBOOK_PATH = ROOT / "docs" / "notebooks" / "gafime_v1_api_reference.ipynb"
COVERAGE_PATH = ROOT / "docs" / "public-api-coverage.md"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gafime_v1_api_reference_generator", GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source_text(notebook: dict[str, object]) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )


def test_v1_reference_is_deterministic_and_substantial() -> None:
    generator = _load_generator()
    notebook = _notebook()

    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == generator.render_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert 60 <= len(notebook["cells"]) <= 85
    reference = notebook["metadata"]["gafime_reference"]
    assert reference == {
        "purpose": "Authoritative current v1 public API reference and cookbook",
        "release_scope": "GAFIME v1 public API",
        "generator": "docs/notebooks/generate_v1_api_reference.py",
        "coverage": "docs/public-api-coverage.md",
        "cells": len(notebook["cells"]),
        "top_level_symbols": len(generator.TOP_LEVEL_PUBLIC_API),
    }
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])


def test_v1_reference_code_cells_compile_and_selected_examples_execute() -> None:
    notebook = _notebook()
    namespace: dict[str, object] = {"__name__": "gafime_v1_reference_smoke"}
    executed_groups: set[str] = set()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compiled = compile(source, f"gafime_v1_api_reference[{index}]", "exec")
        group = cell.get("metadata", {}).get("gafime_test", "syntax")
        if group in {"core", "sklearn", "polars"}:
            exec(compiled, namespace)
            executed_groups.add(group)

    assert executed_groups == {"core", "sklearn", "polars"}


def test_reference_inventory_matches_top_level_exports_and_coverage() -> None:
    generator = _load_generator()
    expected = set(gafime.__all__) | {"__version__"}
    assert set(generator.TOP_LEVEL_PUBLIC_API) == expected
    assert len(generator.TOP_LEVEL_PUBLIC_API) == len(expected)

    notebook_source = _source_text(_notebook())
    coverage = COVERAGE_PATH.read_text(encoding="utf-8")
    for name in generator.TOP_LEVEL_PUBLIC_API:
        assert f"`gafime.{name}`" in notebook_source
        assert f"`gafime.{name}`" in coverage


def test_reference_covers_all_configuration_and_result_fields() -> None:
    generator = _load_generator()
    from gafime.compile.exports import ExportHandles
    from gafime.compile.scenario import (
        ChunkRange,
        ContinuousArityDescriptor,
        ScenarioPlan,
        TimeSeriesDescriptor,
    )
    from gafime.reporting import PermutationResult, StabilityResult

    models = {
        "ComputeBudget": gafime.ComputeBudget,
        "EngineConfig": gafime.EngineConfig,
        "CompileFlags": gafime.CompileFlags,
        "CapabilityValue": gafime.CapabilityValue,
        "BackendCapabilities": gafime.BackendCapabilities,
        "BackendInfo": gafime.BackendInfo,
        "InteractionResult": gafime.InteractionResult,
        "StabilityResult": StabilityResult,
        "PermutationResult": PermutationResult,
        "Decision": gafime.Decision,
        "DiagnosticReport": gafime.DiagnosticReport,
        "DecisionPathCandidate": gafime.DecisionPathCandidate,
        "FamilySignificanceSupport": gafime.FamilySignificanceSupport,
        "FamilyCapability": gafime.FamilyCapability,
        "ExportHandles": ExportHandles,
        "ChunkRange": ChunkRange,
        "ContinuousArityDescriptor": ContinuousArityDescriptor,
        "TimeSeriesDescriptor": TimeSeriesDescriptor,
        "ScenarioPlan": ScenarioPlan,
    }
    documented_source = (
        _source_text(_notebook()) + "\n" + COVERAGE_PATH.read_text(encoding="utf-8")
    )

    for name, model in models.items():
        actual = tuple(field.name for field in fields(model))
        documented = tuple(generator.DOCUMENTED_DATACLASS_FIELDS[name])
        assert documented == actual
        for field_name in actual:
            assert f"`{field_name}`" in documented_source


def _public_members(model: type) -> set[str]:
    public = set()
    for name, value in vars(model).items():
        if name == "__arrow_c_array__" or not name.startswith("_"):
            if isinstance(value, (property, classmethod, staticmethod)) or callable(
                value
            ):
                public.add(name)
    return public


def test_reference_method_inventory_matches_public_class_methods() -> None:
    generator = _load_generator()
    models = {
        "GafimeEngine": gafime.GafimeEngine,
        "NativeCompiledGafime": gafime.NativeCompiledGafime,
        "BackendCapabilities": gafime.BackendCapabilities,
        "DiagnosticReport": gafime.DiagnosticReport,
        "GafimeSelector": gafime.GafimeSelector,
        "GafimeStreamer": gafime.GafimeStreamer,
        "DecisionPathCandidate": gafime.DecisionPathCandidate,
        "FamilyCapability": gafime.FamilyCapability,
    }
    notebook_source = _source_text(_notebook())

    for name, model in models.items():
        expected = set(generator.PUBLIC_METHOD_COVERAGE[name])
        assert _public_members(model) == expected
        for member in expected:
            assert any(
                token in notebook_source
                for token in (f"`{member}`", f".{member}", f"{member}(")
            )


def test_important_public_signatures_are_documented_exactly() -> None:
    assert inspect.signature(gafime.EngineConfig).parameters["precision"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(gafime.ComputeBudget)
        .parameters["max_time_series_candidates"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(gafime.backend_capabilities).parameters["precision"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(gafime.GafimeSelector).parameters["precision"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(gafime.compile).parameters["config"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(gafime.dataload).parameters["config"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(gafime.GafimeStreamer).parameters["precision"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )


def _explicit_docstrings(path: Path) -> dict[str, str | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.get_docstring(node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _missing_public_method_docstrings(path: Path, names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in names:
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name.startswith("_") and member.name != "__arrow_c_array__":
                continue
            if not ast.get_docstring(member):
                missing.append(f"{node.name}.{member.name}")
    return missing


def test_public_python_symbols_have_authored_source_docstrings() -> None:
    expected = {
        "python/gafime/api.py": ("GafimeEngine", "compile"),
        "python/gafime/capabilities.py": (
            "CapabilityValue",
            "BackendCapabilities",
            "backend_capabilities",
        ),
        "python/gafime/compile/flags.py": ("CompileFlags",),
        "python/gafime/compile/api.py": ("compile",),
        "python/gafime/compile/exports.py": ("ExportHandles", "unsupported_export"),
        "python/gafime/compile/scenario.py": (
            "ChunkRange",
            "ContinuousArityDescriptor",
            "ScenarioPlan",
            "TimeSeriesDescriptor",
            "build_scenario_plan",
            "build_scenario_plan_from_shape",
        ),
        "python/gafime/config.py": ("ComputeBudget", "EngineConfig"),
        "python/gafime/dataloader.py": ("dataload",),
        "python/gafime/decision_path.py": (
            "DecisionPathCandidate",
            "decision_path_candidate_from_record",
            "decision_path_candidate_from_result",
            "decision_path_feature_names",
            "describe_decision_path_candidate",
            "evaluate_decision_path_candidate",
            "score_decision_path_candidates",
        ),
        "python/gafime/errors.py": ("GafimeV1Error", "V1UnsupportedError"),
        "python/gafime/families.py": (
            "FamilySignificanceSupport",
            "FamilyCapability",
            "available_families",
            "family_capability",
            "require_family_supported",
        ),
        "python/gafime/io.py": (
            "GafimeStreamer",
            "benchmark_streaming",
            "create_streamer",
        ),
        "python/gafime/reporting/report.py": (
            "BackendInfo",
            "InteractionResult",
            "StabilityResult",
            "PermutationResult",
            "Decision",
            "DiagnosticReport",
            "NativeReportBuilder",
            "NativeContinuousInteractions",
        ),
        "python/gafime/sklearn.py": ("GafimeSelector",),
        "python/gafime/tutorial.py": ("generate_tutorial",),
        "python/gafime/v1_adapter.py": ("NativeCompiledGafime",),
    }
    for relative, names in expected.items():
        docs = _explicit_docstrings(ROOT / relative)
        missing = [name for name in names if not docs.get(name)]
        assert not missing, f"{relative} has undocumented public symbols: {missing}"
        missing_methods = _missing_public_method_docstrings(ROOT / relative, names)
        assert not missing_methods, (
            f"{relative} has undocumented public methods: {missing_methods}"
        )

    subfunctions = ast.parse(
        (ROOT / "python" / "gafime" / "subfunctions.py").read_text(encoding="utf-8")
    )
    assert ast.get_docstring(subfunctions)


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _local_links(markdown: str) -> list[str]:
    return [
        target.split("#", 1)[0]
        for target in _MARKDOWN_LINK.findall(markdown)
        if target and not target.startswith(("http://", "https://", "mailto:", "#"))
    ]


def test_documentation_hierarchy_and_local_links_are_valid() -> None:
    hierarchy = {
        ROOT / "README.md": "docs/notebooks/gafime_v1_api_reference.ipynb",
        ROOT / "USAGE.md": "docs/notebooks/gafime_v1_api_reference.ipynb",
        ROOT / "python" / "gafime" / "tutorial.py": "gafime_v1_api_reference.ipynb",
    }
    for path, required in hierarchy.items():
        assert required in path.read_text(encoding="utf-8")

    markdown_sources = {
        ROOT / "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        ROOT / "USAGE.md": (ROOT / "USAGE.md").read_text(encoding="utf-8"),
        ROOT / "docs" / "README.md": (ROOT / "docs" / "README.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "docs" / "releases" / "README.md": (
            ROOT / "docs" / "releases" / "README.md"
        ).read_text(encoding="utf-8"),
        ROOT / "docs" / "releases" / "STATUS.md": (
            ROOT / "docs" / "releases" / "STATUS.md"
        ).read_text(encoding="utf-8"),
        COVERAGE_PATH: COVERAGE_PATH.read_text(encoding="utf-8"),
    }
    notebook_markdown = "\n".join(
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "markdown"
    )
    markdown_sources[NOTEBOOK_PATH] = notebook_markdown

    for source_path, markdown in markdown_sources.items():
        for target in _local_links(markdown):
            resolved = (source_path.parent / target).resolve()
            assert resolved.exists(), f"broken local link in {source_path}: {target}"


def test_reference_records_current_non_api_boundaries() -> None:
    source = " ".join(_source_text(_notebook()).split())
    for required in (
        "Pearson correlation squared, clamped to `[0, 1]`",
        "max_generated_features",
        "n_jobs",
        "thread-affine",
        "There is no context-manager or `run()` API",
        "raw Arrow table/stream is not a top-level `analyze()` input",
        "advanced compatibility/native boundary",
        "RT/OptiX remains experimental and local-only",
        "not a universal performance claim",
        "validated at construction and through `set_params()`",
        "validated before the reader evaluates the source row count",
        "validated before the file is opened",
    ):
        assert required in source
