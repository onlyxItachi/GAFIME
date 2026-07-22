#!/usr/bin/env python3
"""Validate current release-facing docs, skills, and practice notebook."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
SKILL_NAMES = {
    "benchmark-vs-manual",
    "build-pipeline",
    "check-install",
    "dataset-profiler",
    "interpret-results",
    "platform-detect",
    "time-series-setup",
    "troubleshoot-backend",
    "validate-features",
}
REMOVED_GUIDANCE = {
    "enable_discrete_functions",
    "discrete_mode",
    "discrete_ranking",
    "gafime_discrete_selection_adaptive_cuda",
    "benchmark_v045_native_spine.py",
    "C++ Core",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_skills() -> None:
    skill_root = ROOT / ".claude" / "skills"
    actual = {path.parent.name for path in skill_root.glob("*/SKILL.md")}
    missing = sorted(SKILL_NAMES - actual)
    _require(not missing, f"required release-facing skills are missing: {missing}")
    texts = []
    for path in sorted(skill_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        texts.append(text)
        if path.suffix == ".py":
            compile(text, str(path), "exec")
    combined = "\n".join(texts)
    for phrase in REMOVED_GUIDANCE:
        _require(phrase not in combined, f"support skills contain removed guidance: {phrase}")
    combined_lower = combined.lower()
    for phrase in (
        "backend_capabilities",
        "decision-path permutation significance",
        "gafime v1",
        "candidate-row",
    ):
        _require(phrase in combined_lower, f"support skills are missing current guidance: {phrase}")


def _validate_notebook() -> None:
    generator_path = ROOT / "python" / "gafime" / "tutorial.py"
    tracked_path = ROOT / "docs" / "notebooks" / "gafime_tutorial.ipynb"
    module = _load_module(generator_path, "gafime_release_tutorial")
    with tempfile.TemporaryDirectory(prefix="gafime-tutorial-contract-") as temp_dir:
        generated_path = Path(module.generate_tutorial(str(Path(temp_dir) / "tutorial.ipynb")))
        _require(
            generated_path.read_bytes() == tracked_path.read_bytes(),
            "tracked practice notebook differs from generate_tutorial output",
        )

    notebook = json.loads(tracked_path.read_text(encoding="utf-8"))
    reference = notebook.get("metadata", {}).get("gafime_reference", {})
    _require(reference.get("release_scope") == "GAFIME v1 public API", "notebook scope is stale")
    _require(reference.get("generator") == "python/gafime/tutorial.py", "notebook generator is undisclosed")
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for token in (
        "backend_capabilities('auto', probe=True)",
        "CompileFlags(plan=True, graph=False, export=False)",
        "enable_time_series_functions=True",
        "enable_decision_path_functions=True",
        "permutation_tests=0",
        "available_families",
        "GafimeSelector",
    ):
        _require(token in code, f"practice notebook is missing v1 example: {token}")


def _validate_pipeline_generator() -> None:
    path = ROOT / ".claude" / "skills" / "build-pipeline" / "scripts" / "generate_pipeline.py"
    module = _load_module(path, "gafime_release_pipeline_generator")
    classification = module.generate_pipeline_script("classification", model="auto")
    regression = module.generate_pipeline_script("regression", model="auto")
    compile(classification, "generated-classification-pipeline.py", "exec")
    compile(regression, "generated-regression-pipeline.py", "exec")
    _require("LogisticRegression" in classification, "classification auto model is not sklearn-only")
    _require("Ridge" in regression, "regression auto model is not sklearn-only")


def _validate_release_docs() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(project["project"]["version"])
    release_note = ROOT / "docs" / "releases" / f"v{version}.md"
    runbook = ROOT / "docs" / "releases" / "release-operations.md"
    _require(release_note.is_file(), f"missing release note for {version}")
    _require(runbook.is_file(), "missing release operations runbook")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for link in (
        f"docs/releases/v{version}.md",
        "docs/releases/release-operations.md",
        "docs/capabilities.md",
        "docs/eager-resident-compiled-execution.md",
        "docs/notebooks/gafime_tutorial.ipynb",
    ):
        _require(link in readme, f"README does not expose {link}")

    note_text = release_note.read_text(encoding="utf-8")
    _require("release-operations.md" in note_text, "release note does not link the runbook")
    runbook_text = runbook.read_text(encoding="utf-8")
    for token in (
        "publish_pypi_core=false",
        "publish_pypi_cuda=false",
        "publish_pypi_rocm=false",
        "publish_github_release=false",
        "build_cuda_rt_payload=false",
        "allow_matching_existing_pypi_files=false",
        "check_pypi_collisions=true",
        "publish_pypi_core=true",
        "publish_pypi_cuda=true",
        "publish_pypi_rocm=true",
        "publish_github_release=true",
        "allow_matching_existing_pypi_files=true",
        "SHA-256",
        "11 artifacts",
    ):
        _require(token in runbook_text, f"release runbook is missing {token}")

    workflow = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
        encoding="utf-8"
    )
    for input_name in (
        "publish_pypi_core",
        "publish_pypi_cuda",
        "publish_pypi_rocm",
        "publish_github_release",
        "build_cuda_rt_payload",
        "allow_matching_existing_pypi_files",
        "check_pypi_collisions",
    ):
        _require(f"      {input_name}:" in workflow, f"runbook input is absent from workflow: {input_name}")


def main() -> None:
    _validate_skills()
    _validate_notebook()
    _validate_pipeline_generator()
    _validate_release_docs()
    print("release-facing docs, skills, notebook, and runbook verified")


if __name__ == "__main__":
    main()
