#!/usr/bin/env python3
"""Validate current release-facing docs, skills, and practice notebook."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
from release_manifest import load_release_manifest, render_release_matrix  # noqa: E402
from release_version import validate_project_versions  # noqa: E402


RELEASE_MANIFEST = load_release_manifest(ROOT)
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
    "gafime-cuda-rt",
    "gafime-rocm-bundled",
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
        _require(
            phrase not in combined, f"support skills contain removed guidance: {phrase}"
        )
    combined_lower = combined.lower()
    for phrase in (
        "backend_capabilities",
        "decision-path permutation significance",
        "gafime v1",
        "candidate-row",
        "conditional on selection",
        "does not correct selection bias",
    ):
        _require(
            phrase in combined_lower,
            f"support skills are missing current guidance: {phrase}",
        )

    interpreter_path = (
        skill_root / "interpret-results" / "scripts" / "explain_report.py"
    )
    interpreter = _load_module(interpreter_path, "gafime_interpret_results")
    explained = interpreter.explain_report(
        {
            "interactions": [
                {
                    "candidate_id": "interaction:0,1",
                    "family": "interaction",
                    "combo": [0, 1],
                    "metrics": {"pearson": 0.8},
                }
            ],
            "stability": [
                {
                    "candidate_id": "interaction:0,1",
                    "metrics_std": {"pearson": 0.02},
                }
            ],
        }
    )
    pearson = explained["interactions"][0]["metrics"]["pearson"]
    _require(
        pearson["stability"] == "low conditional variability"
        and pearson["stability_scope"]
        == "conditional on selection using the same rows",
        "interpret-results must label bootstrap variability conditionally",
    )

    health_path = skill_root / "check-install" / "scripts" / "health_check.py"
    health = subprocess.run(
        [sys.executable, str(health_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    health_output = (health.stdout + health.stderr).strip()
    _require(
        health.returncode == 0,
        f"check-install health check failed: {health_output[-1000:]}",
    )
    _require(
        "[PASS] Family capability contract:" in health_output
        and "decision_path_permutation=true" in health_output,
        "check-install health check does not admit decision-path permutation support",
    )


def _validate_notebook() -> None:
    generator_path = ROOT / "python" / "gafime" / "tutorial.py"
    tracked_path = ROOT / "docs" / "notebooks" / "gafime_tutorial.ipynb"
    module = _load_module(generator_path, "gafime_release_tutorial")
    with tempfile.TemporaryDirectory(prefix="gafime-tutorial-contract-") as temp_dir:
        generated_path = Path(
            module.generate_tutorial(str(Path(temp_dir) / "tutorial.ipynb"))
        )
        _require(
            generated_path.read_bytes() == tracked_path.read_bytes(),
            "tracked practice notebook differs from generate_tutorial output",
        )

    notebook = json.loads(tracked_path.read_text(encoding="utf-8"))
    reference = notebook.get("metadata", {}).get("gafime_reference", {})
    _require(
        reference.get("release_scope") == "GAFIME v1 public API",
        "notebook scope is stale",
    )
    _require(
        reference.get("generator") == "python/gafime/tutorial.py",
        "notebook generator is undisclosed",
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for token in (
        "backend_capabilities('auto', probe=True, precision='mixed')",
        "CompileFlags(export=True)",
        "precision='mixed'",
        "enable_time_series_functions=True",
        "enable_decision_path_functions=True",
        "permutation_tests=0",
        "available_families",
        "GafimeSelector",
    ):
        _require(token in code, f"practice notebook is missing v1 example: {token}")


def _validate_v1_api_reference() -> None:
    generator_path = ROOT / "docs" / "notebooks" / "generate_v1_api_reference.py"
    notebook_path = ROOT / "docs" / "notebooks" / "gafime_v1_api_reference.ipynb"
    coverage_path = ROOT / "docs" / "public-api-coverage.md"
    for path, description in (
        (generator_path, "v1 API reference generator"),
        (notebook_path, "generated v1 API reference"),
        (coverage_path, "public API coverage inventory"),
    ):
        _require(path.is_file(), f"missing {description}: {path.relative_to(ROOT)}")

    generator = _load_module(generator_path, "gafime_v1_api_reference_generator")
    _require(
        notebook_path.read_text(encoding="utf-8") == generator.render_notebook(),
        "tracked v1 API reference differs from its deterministic generator",
    )
    parity = subprocess.run(
        [sys.executable, str(generator_path), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    _require(
        parity.returncode == 0,
        "v1 API reference generator check failed: "
        f"{(parity.stderr or parity.stdout).strip()}",
    )

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    reference = notebook.get("metadata", {}).get("gafime_reference", {})
    _require(
        reference.get("release_scope") == "GAFIME v1 public API",
        "v1 API reference scope is stale",
    )
    _require(
        reference.get("generator") == "docs/notebooks/generate_v1_api_reference.py",
        "v1 API reference generator is undisclosed",
    )
    _require(
        reference.get("coverage") == "docs/public-api-coverage.md",
        "v1 API reference does not identify its coverage inventory",
    )

    coverage = coverage_path.read_text(encoding="utf-8")
    _require(
        "[v1 API reference](notebooks/gafime_v1_api_reference.ipynb)" in coverage,
        "public API coverage inventory does not link the authoritative reference",
    )

    hierarchy = {
        ROOT / "README.md": "docs/notebooks/gafime_v1_api_reference.ipynb",
        ROOT / "USAGE.md": "docs/notebooks/gafime_v1_api_reference.ipynb",
        ROOT / "python" / "gafime" / "tutorial.py": ("gafime_v1_api_reference.ipynb"),
        ROOT / "docs" / "notebooks" / "gafime_tutorial.ipynb": (
            "gafime_v1_api_reference.ipynb"
        ),
    }
    for path, target in hierarchy.items():
        _require(
            target in path.read_text(encoding="utf-8"),
            f"{path.relative_to(ROOT)} does not link the current v1 API reference",
        )


def _validate_pipeline_generator() -> None:
    path = (
        ROOT
        / ".claude"
        / "skills"
        / "build-pipeline"
        / "scripts"
        / "generate_pipeline.py"
    )
    module = _load_module(path, "gafime_release_pipeline_generator")
    classification = module.generate_pipeline_script("classification", model="auto")
    regression = module.generate_pipeline_script("regression", model="auto")
    compile(classification, "generated-classification-pipeline.py", "exec")
    compile(regression, "generated-regression-pipeline.py", "exec")
    _require(
        "LogisticRegression" in classification,
        "classification auto model is not sklearn-only",
    )
    _require("Ridge" in regression, "regression auto model is not sklearn-only")


def _documented_gafime_commands(path: Path) -> list[str]:
    commands = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or "->" in stripped:
            continue
        if stripped == "gafime" or stripped.startswith("gafime "):
            commands.append(stripped)
    return commands


def _validate_documented_cli_commands() -> None:
    paths = [ROOT / "README.md", ROOT / "USAGE.md", ROOT / "BUILD.md"]
    paths.extend(sorted((ROOT / "docs").rglob("*.md")))
    commands = [
        (path, command)
        for path in paths
        for command in _documented_gafime_commands(path)
    ]
    _require(commands, "release-facing docs contain no GAFIME CLI smoke commands")

    for path, command in commands:
        args = shlex.split(command)
        result = subprocess.run(
            [sys.executable, "-m", "gafime.cli", *args[1:]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[-500:]
        _require(
            result.returncode in {0, 1},
            f"documented CLI command does not parse in {path.relative_to(ROOT)}: "
            f"{command!r} (exit={result.returncode}, output={detail!r})",
        )


def _validate_release_docs() -> None:
    release = validate_project_versions(ROOT)
    version = release.pep440
    release_note = ROOT / release.release_note
    runbook = ROOT / "docs" / "releases" / "release-operations.md"
    release_matrix = ROOT / "docs" / "releases" / "release-artifact-matrix.md"
    pypi_status = ROOT / ".github" / "scripts" / "check_pypi_release_status.py"
    release_bundle = ROOT / ".github" / "scripts" / "release_bundle.py"
    build_workflow_path = ROOT / ".github" / "workflows" / "build_wheels.yml"
    publish_workflow_path = ROOT / ".github" / "workflows" / "publish_release.yml"
    _require(release_note.is_file(), f"missing release note for {version}")
    _require(runbook.is_file(), "missing release operations runbook")
    _require(release_matrix.is_file(), "missing generated release artifact matrix")
    _require(pypi_status.is_file(), "missing PyPI release-status verifier")
    _require(release_bundle.is_file(), "missing immutable release-bundle verifier")
    _require(build_workflow_path.is_file(), "missing build/freeze workflow")
    _require(publish_workflow_path.is_file(), "missing frozen-bundle publisher")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    release_index = (ROOT / "docs" / "releases" / "README.md").read_text(
        encoding="utf-8"
    )
    release_status = (ROOT / "docs" / "releases" / "STATUS.md").read_text(
        encoding="utf-8"
    )
    for link in (
        "docs/README.md",
        "docs/releases/STATUS.md",
        "docs/releases/README.md",
        "docs/capabilities.md",
        "docs/eager-resident-compiled-execution.md",
        "docs/notebooks/gafime_tutorial.ipynb",
    ):
        _require(link in readme, f"README does not route to {link}")
    for link in (
        "contract.md",
        "abi-evolution.md",
        "releases/release-operations.md",
        "releases/release-artifact-matrix.md",
        "security/threat-model.md",
    ):
        _require(link in docs_index, f"docs/README.md does not route to {link}")
    _require(
        Path(release.release_note).name in release_index,
        "release index does not expose the current version record",
    )
    _require(
        "mutable operational status, not an immutable historical release record"
        in " ".join(release_status.split()),
        "release STATUS does not disclose its mutable role",
    )
    for token in (
        'precision="mixed"',
        "Rust Core/SIMD",
        "`fp32`, `mixed`, `fp64`",
        "`fp32` only",
        "docs/precision-contract.md",
    ):
        _require(token in readme, f"README is missing precision guidance: {token}")
    usage = (ROOT / "USAGE.md").read_text(encoding="utf-8")
    for token in (
        'precision="mixed"',
        "`fp32`: fp32 ingest/storage",
        "`mixed` (default)",
        "`fp64`: fp64 ingest/storage",
        "float32+fast -> fp32",
        'backend="metal"',
    ):
        _require(token in usage, f"USAGE is missing precision guidance: {token}")
    build_guide = (ROOT / "BUILD.md").read_text(encoding="utf-8")
    for token in (
        "Windows ARM64 uses an ARM64 Python 3.11",
        "workflow host while cibuildwheel",
        "including CPython 3.10",
        "`pythonarm64` NuGet packages",
    ):
        _require(
            token in build_guide,
            f"BUILD is missing full Windows ARM64 wheel coverage: {token}",
        )

    note_text = release_note.read_text(encoding="utf-8")
    _require(
        "release-operations.md" in note_text, "release note does not link the runbook"
    )
    for token in ("## Deliberate Non-Claims", "overflowed before normalization"):
        _require(
            token in note_text, f"release note is missing evidence boundary: {token}"
        )
    for token in (
        "## Precision Profiles",
        "`EngineConfig(precision=...)`",
        "genuine lane-wide fp32",
        "No precision-specific package",
    ):
        _require(
            token in note_text,
            f"release note is missing precision capability: {token}",
        )
    for token in (
        "Windows ARM64 contributes",
        "five dedicated wheels",
        "including CPython 3.10",
        "`pythonarm64` NuGet packages",
    ):
        _require(
            token in note_text,
            f"release note is missing full Windows ARM64 wheel coverage: {token}",
        )
    for token in (
        release.tag,
        release.semver,
        release.pep440,
        "Semantic Versioning",
        "PEP 440",
    ):
        _require(
            token in note_text,
            f"release note is missing version-policy identity: {token}",
        )
    runbook_text = runbook.read_text(encoding="utf-8")
    normalized_runbook = " ".join(runbook_text.split())
    matrix_text = release_matrix.read_text(encoding="utf-8")
    _require(
        matrix_text == render_release_matrix(RELEASE_MANIFEST),
        "release artifact matrix differs from .github/release-artifacts.json",
    )
    for token in (
        "build_wheels.yml",
        "publish_release.yml",
        "build_run_id=<build-run-id>",
        "release_tag=v<semver>",
        "allow_matching_existing_pypi_files=false",
        "release_version.py --check-project",
        "v<semver>",
        "<pep440>",
        "allow_matching_existing_pypi_files=true",
        "SHA-256",
        ".github/release-artifacts.json",
        "release-artifact-matrix.md",
        "rocm-wheel-policy-report.json",
        "libamdhip64.so.7",
        "Every Core, CUDA, and ROCm wheel contains `fp32`, `mixed`, and `fp64`",
        "exact profile capability mask",
        "compressed wheel and uncompressed",
        "Core -> CUDA and ROCm -> public exact-version installs -> GitHub Release",
        "must never build, repair, retag, rename, or otherwise mutate a package",
        "RT/OptiX is locally buildable through CMake only",
        "## Abandoned Partial Publication",
        "--expect-missing gafime==1.0.0b1",
        "--expect-yanked gafime-cuda==1.0.0b1",
        "--expect-yanked gafime-rocm==1.0.0b1",
        "PyPI's release-yanking guidance",
        "PEP 592",
    ):
        _require(
            token in runbook_text or token in normalized_runbook,
            f"release runbook is missing {token}",
        )

    normal_publication = runbook_text.split("## Normal Publication", 1)[1].split(
        "## Hash-Matched Recovery", 1
    )[0]
    tag_creation = normal_publication.find("create `v<semver>`")
    _require(tag_creation >= 0, "normal publication is missing canonical tag creation")
    for prerequisite in (
        "Confirm all three Trusted Publisher entries name `publish_release.yml`",
        "environment `pypi`",
        "retired `build_wheels.yml` entries",
        "python .github/scripts/check_pypi_release_status.py",
        "--expect-missing gafime==1.0.0b1",
        "--expect-yanked gafime-cuda==1.0.0b1",
        "--expect-yanked gafime-rocm==1.0.0b1",
        '--reason-contains "matching gafime==1.0.0b1 Core was not published"',
    ):
        position = normal_publication.find(prerequisite)
        _require(
            0 <= position < tag_creation,
            f"normal publication must verify {prerequisite!r} before tag creation",
        )
    _require(
        "After one successful publication, remove or disable the old entries"
        not in normalized_runbook,
        "release runbook must not retain the retired post-publication publisher migration",
    )

    pypi_status_result = subprocess.run(
        [sys.executable, str(pypi_status), "--self-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        pypi_status_result.returncode == 0,
        "PyPI release-status verifier self-test failed: "
        f"{(pypi_status_result.stderr or pypi_status_result.stdout).strip()}",
    )
    bundle_result = subprocess.run(
        [sys.executable, str(release_bundle), "--self-test"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        bundle_result.returncode == 0,
        "immutable release-bundle verifier self-test failed: "
        f"{(bundle_result.stderr or bundle_result.stdout).strip()}",
    )

    abandoned_note = ROOT / "docs" / "releases" / "v1.0.0b1.md"
    abandoned_text = abandoned_note.read_text(encoding="utf-8")
    for token in (
        "## Resolver Safety",
        "gafime-cuda==1.0.0b1",
        "gafime-rocm==1.0.0b1",
        "must be yanked",
        "immutable historical records",
    ):
        _require(
            token in abandoned_text,
            f"aborted b1 release note is missing resolver-safety evidence: {token}",
        )
    for token in (
        "Historical compact tags",
        "mixed naming",
        "intentional historical preservation",
    ):
        _require(
            token in note_text,
            f"release note is missing historical-version policy: {token}",
        )

    build_workflow = build_workflow_path.read_text(encoding="utf-8")
    publish_workflow = publish_workflow_path.read_text(encoding="utf-8")
    _require(
        "      core_wheel_build_tag:" in build_workflow,
        "build workflow must expose only the pre-freeze Core build-tag input",
    )
    for token in (
        "--machine-code-evidence-dir",
        "--verify-wheel-evidence dist",
        "Verify requested retagged Core wheels install as exact archives",
    ):
        _require(
            token in build_workflow,
            f"build/freeze workflow is missing artifact-profile proof: {token}",
        )
    static_report = (
        ROOT / "tests" / "release_measure" / "gpu_static_kernel_report.py"
    ).read_text(encoding="utf-8")
    _require(
        "wheel_sha256=" in static_report and "native_sha256=" in static_report,
        "wheel evidence must record both wheel and native-member SHA-256",
    )
    for forbidden in (
        "pypa/gh-action-pypi-publish",
        "softprops/action-gh-release",
        "publish_pypi_core:",
        "publish_pypi_cuda:",
        "publish_pypi_rocm:",
        "publish_github_release:",
        "gafime-cuda-rt",
        "gafime-rocm-bundled",
    ):
        _require(
            forbidden not in build_workflow,
            f"build/freeze workflow contains publication or retired path: {forbidden}",
        )
    for input_name in (
        "build_run_id",
        "release_tag",
        "allow_matching_existing_pypi_files",
    ):
        _require(
            f"      {input_name}:" in publish_workflow,
            f"publisher input is absent from workflow: {input_name}",
        )
    for forbidden_builder in (
        "maturin-action",
        "python -m cibuildwheel",
        "python -m build",
        "cargo build",
        "nvcc",
        "hipcc",
    ):
        _require(
            forbidden_builder not in publish_workflow,
            f"publisher must not rebuild frozen artifacts: {forbidden_builder}",
        )
    publication_prefix = publish_workflow.split(
        "\n  verify_public_core_and_cuda:\n", 1
    )[0]
    github_release = publish_workflow.split("\n  publish_github_release:\n", 1)[1]
    _require(
        "cibuildwheel" not in publication_prefix
        and "cibuildwheel" not in github_release,
        "frozen-bundle publication jobs must not invoke or install wheel builders",
    )
    windows_arm_public = publish_workflow.split(
        "\n  verify_public_windows_arm_core:\n", 1
    )[1].split("\n  verify_public_rocm_install:\n", 1)[0]
    for token in (
        '"3.10"',
        "cp310-win_arm64",
        'python-version: "3.11"',
        "cibuildwheel==3.4.1",
        "provision_windows_arm64_python.py",
        "--venv ",
        "$env:TARGET_PYTHON",
    ):
        _require(
            token in windows_arm_public,
            f"public Windows ARM64 validation is missing target-runtime proof: {token}",
        )
    for token in (
        "name: release-bundle",
        "release_bundle.py verify",
        "needs: [publication_preflight, publish_pypi_core]",
        "verify_public_core_and_cuda",
        "verify_public_windows_arm_core",
        "verify_public_rocm_install",
        "Publish GitHub Release after public installation",
    ):
        _require(
            token in publish_workflow,
            f"publisher is missing immutable ordered-publication rule: {token}",
        )


def main() -> None:
    _validate_skills()
    _validate_notebook()
    _validate_v1_api_reference()
    _validate_pipeline_generator()
    _validate_documented_cli_commands()
    _validate_release_docs()
    print("release-facing docs, skills, notebook, and runbook verified")


if __name__ == "__main__":
    main()
