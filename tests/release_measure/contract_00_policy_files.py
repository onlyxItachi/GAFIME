#!/usr/bin/env python3
from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CONTRACT_SECTIONS = (
    "## Repository Layout",
    "## Kernel And Orchestration Layout",
    "## Permitted Source Extensions",
    "## Compiler Ownership",
    "## Backend Ownership",
    "## Forbidden Cross-Boundary Calls",
    "## Rust Safety",
    "## ABI Contract",
    "## Numerical Policy",
    "## Feature Generation Verification",
    "## Release Version Policy",
    "## PR, Main, And Release Gates",
    "## Regression Policy",
    "## Migration Rules",
)
AGENT_ONLY_SECTION = "## Delegated Agent Coordination"
HANDOFF_ROUTING_SECTION = "## Context And Handoff Routing"
TRANSIENT_AGENT_MARKERS = (
    "Snapshot date:",
    "Last verified branch state:",
    "## Clear-Recovery Handoff Snapshot",
    "## Performance Hardening Continuation",
    "## Correctness Boundary Hardening Follow-up",
    "## Eager Path Pre-Release Hardening",
    "## PR #",
    "## ROCm Wheel Policy Handoff",
    "## v1.0.0b1 Release-Artifact Repair Handoff",
)
FORBIDDEN_ROOT_AGENT_ARTIFACTS = (
    "AGENT_COMMS.md",
    "AGENT_COMMS_ARCHIVE.md",
    "codex.md",
    "plan.md",
)
FORBIDDEN_AGENT_IGNORE_PATTERNS = (
    "AGENT_COMMS*.md",
    "AGENT_COMMS_ARCHIVE*.md",
    "codex.md",
    "plan.md",
)


def normalized_agent_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 3 and lines[2].startswith("This file mirrors `"):
        lines[2] = (
            "This file mirrors `<mirror>`. Keep both files synchronized except for agent-specific notes that are explicitly needed."
        )
    if path.name == "AGENT.md" and AGENT_ONLY_SECTION in lines:
        start = lines.index(AGENT_ONLY_SECTION)
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("## ")
            ),
            len(lines),
        )
        del lines[start:end]
        while start < len(lines) and not lines[start].strip():
            del lines[start]
    return "\n".join(lines)


def _validate_protected_branch_triggers(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        trigger_start = lines.index("on:")
    except ValueError as exc:
        raise AssertionError(
            f"{path.relative_to(ROOT)} has no workflow triggers"
        ) from exc

    trigger_end = next(
        (
            index
            for index in range(trigger_start + 1, len(lines))
            if lines[index] and not lines[index][0].isspace()
        ),
        len(lines),
    )
    trigger_lines = lines[trigger_start + 1 : trigger_end]
    events = {
        line[2:-1]
        for line in trigger_lines
        if line.startswith("  ") and not line.startswith("   ") and line.endswith(":")
    }
    expected_events = {"pull_request", "push", "workflow_dispatch"}
    if events != expected_events:
        raise AssertionError(
            f"{path.relative_to(ROOT)} must retain exactly the protected-branch "
            f"trigger events {sorted(expected_events)}; found {sorted(events)}"
        )

    pull_request_start = trigger_lines.index("  pull_request:")
    pull_request_end = next(
        (
            index
            for index in range(pull_request_start + 1, len(trigger_lines))
            if trigger_lines[index].startswith("  ")
            and not trigger_lines[index].startswith("   ")
        ),
        len(trigger_lines),
    )
    pull_request_lines = [
        line
        for line in trigger_lines[pull_request_start + 1 : pull_request_end]
        if line
    ]
    if pull_request_lines:
        if pull_request_lines[0] != "    branches:" or any(
            not line.startswith("      - ") for line in pull_request_lines[1:]
        ):
            raise AssertionError(
                f"{path.relative_to(ROOT)} pull_request trigger must be unfiltered "
                "or declare protected branches"
            )
        pull_request_branches = tuple(
            line.removeprefix("      - ").strip().strip("'\"")
            for line in pull_request_lines[1:]
        )
        if pull_request_branches != ("main", "release/v*"):
            raise AssertionError(
                f"{path.relative_to(ROOT)} pull_request trigger must cover "
                "main and release/v* in that order"
            )

    push_start = trigger_lines.index("  push:")
    push_end = next(
        (
            index
            for index in range(push_start + 1, len(trigger_lines))
            if trigger_lines[index].startswith("  ")
            and not trigger_lines[index].startswith("   ")
        ),
        len(trigger_lines),
    )
    push_lines = [line for line in trigger_lines[push_start + 1 : push_end] if line]
    if not push_lines or push_lines[0] != "    branches:":
        raise AssertionError(
            f"{path.relative_to(ROOT)} push trigger must declare protected branches"
        )
    branch_lines = push_lines[1:]
    if any(not line.startswith("      - ") for line in branch_lines):
        raise AssertionError(
            f"{path.relative_to(ROOT)} push trigger has unsupported branch policy"
        )
    branches = tuple(
        line.removeprefix("      - ").strip().strip("'\"") for line in branch_lines
    )
    expected_branches = ("main", "release/v*")
    if branches != expected_branches:
        raise AssertionError(
            f"{path.relative_to(ROOT)} push branches must be "
            f"{expected_branches}; found {branches}"
        )


def main() -> None:
    contract = ROOT / "docs" / "contract.md"
    claude = ROOT / "CLAUDE.md"
    agent = ROOT / "AGENT.md"
    contributing = ROOT / "CONTRIBUTING.md"
    build_doc = ROOT / "BUILD.md"
    release_branches = ROOT / "docs" / "releases" / "release-branches.md"
    workflow = ROOT / ".github" / "workflows" / "v1_contract_validation.yml"
    release_workflow = ROOT / ".github" / "workflows" / "build_wheels.yml"
    native_workflow = ROOT / ".github" / "workflows" / "native_platform_validation.yml"
    clippy_config = ROOT / "clippy.toml"
    cargo_manifest = ROOT / "Cargo.toml"
    rust_release_evidence = (
        ROOT / "docs" / "evidence" / "rust-1.97.1-release-compiler.md"
    )
    architecture_gate = ROOT / "tests" / "release_measure" / "v1_architecture_gate.py"
    release_version = ROOT / ".github" / "scripts" / "release_version.py"
    crate_manifests = tuple(sorted((ROOT / "crates").glob("*/Cargo.toml")))
    gitignore = ROOT / ".gitignore"

    for path in (
        contract,
        claude,
        agent,
        contributing,
        build_doc,
        release_branches,
        gitignore,
        workflow,
        release_workflow,
        native_workflow,
        clippy_config,
        cargo_manifest,
        rust_release_evidence,
        architecture_gate,
        release_version,
        *crate_manifests,
    ):
        if not path.exists():
            raise AssertionError(
                f"required contract artifact is missing: {path.relative_to(ROOT)}"
            )

    stale_root_artifacts = [
        name for name in FORBIDDEN_ROOT_AGENT_ARTIFACTS if (ROOT / name).exists()
    ]
    if stale_root_artifacts:
        raise AssertionError(
            f"repo root contains obsolete agent artifacts: {stale_root_artifacts}"
        )
    ignored_paths = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    hidden_agent_artifacts = sorted(
        ignored_paths.intersection(FORBIDDEN_AGENT_IGNORE_PATTERNS)
    )
    if hidden_agent_artifacts:
        raise AssertionError(
            "obsolete root agent artifacts must not be hidden by .gitignore: "
            f"{hidden_agent_artifacts}"
        )

    contract_text = contract.read_text(encoding="utf-8")
    for section in REQUIRED_CONTRACT_SECTIONS:
        if section not in contract_text:
            raise AssertionError(f"docs/contract.md missing section: {section}")
    for phrase in (
        "Standard CUDA payloads compile only `precision_kernels.cu` and `precision_launcher.cu`",
        "Standard ROCm payloads compile both `kernels.hip` and `launcher.hip`",
        "GPU payload staging and release packaging must source backend files from this root `src/` layout",
        "Packaging must not reintroduce `gpu/`, crate-local native source homes",
        "CPU fixed-bin mutual information is the CPU parity path for the GPU-compatible MI approximation",
        "bundle no ROCm userspace, carry no RPATH or RUNPATH",
        "PyPI receives the matching source distribution",
        "There is no bundled-runtime\nROCm distribution policy",
        "There is no RT distribution identity",
        "dynamically requires\nthe system CUDA runtime",
        "must not vendor `libcudart`, `cudart64`, or\n`nvcudart`",
        "Apple Silicon Metal is embedded only in the `gafime` macOS arm64 core wheel",
        "Core and payload wheels use dedicated CPython ABIs",
        "Core must not depend on CUDA or ROCm payload distributions",
        "Build and publication workflows remain separate",
        "Publication order is Core, CUDA/ROCm, public exact-version install verification",
        "Artifact counts are derived from the per-CPython/platform",
        "must not enter a wheel, sdist,\nworkflow artifact, cache artifact, or GitHub Release",
        "The Cargo workspace version is the canonical repository release input",
        "`.github/scripts/release_version.py` is the",
        "Prerelease classification is parser-derived",
    ):
        if phrase not in contract_text:
            raise AssertionError(
                f"docs/contract.md missing GPU packaging rule: {phrase}"
            )

    agent_text = agent.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8")
    contributing_text = contributing.read_text(encoding="utf-8")
    governance_phrases = (
        "`main` remains protected",
        "accepts tracked changes only through a pull request",
        "required GitHub approving-review count is zero",
        "independent human approval is not required",
        "current-head AI Review Record",
        "A `COMMENTED` review is valid review evidence",
        "all review conversations resolved",
        "model, role, exact reviewed commit SHA, verdict, and findings",
        "later head commit invalidates the record",
        "base change invalidates the merge-commit CI evidence",
        "merge-blocking verdict or unresolved blocking finding prevents merge",
        "Intermediate PR commits do not need to be green",
        "GitHub's current PR merge commit",
        "resulting commit on `main`",
        "`@onlyxItachi` is the sole final merge authority",
    )
    for path, policy_text in (
        (contract, contract_text),
        (agent, agent_text),
        (claude, claude_text),
        (contributing, contributing_text),
    ):
        for phrase in governance_phrases:
            if phrase not in policy_text:
                raise AssertionError(
                    f"{path.relative_to(ROOT)} missing review governance: {phrase}"
                )
        if "Every PR and every commit" in policy_text:
            raise AssertionError(
                f"{path.relative_to(ROOT)} retains obsolete every-commit CI policy"
            )
    if AGENT_ONLY_SECTION not in agent_text or AGENT_ONLY_SECTION in claude_text:
        raise AssertionError("Codex delegation rules must exist only in AGENT.md")
    for path, policy_text in ((agent, agent_text), (claude, claude_text)):
        if HANDOFF_ROUTING_SECTION not in policy_text:
            raise AssertionError(
                f"{path.name} must route transient status outside stable policy"
            )
        stale_markers = [
            marker for marker in TRANSIENT_AGENT_MARKERS if marker in policy_text
        ]
        if stale_markers:
            raise AssertionError(
                f"{path.name} contains transient handoff state: {stale_markers}"
            )
        if "## Release Version Policy" not in policy_text:
            raise AssertionError(
                f"{path.name} must define the permanent release-version policy"
            )
        for retired_identity in ("gafime-cuda-rt", "gafime-rocm-bundled"):
            if retired_identity in policy_text:
                raise AssertionError(
                    f"{path.name} contains retired distribution identity: "
                    f"{retired_identity}"
                )
    for required_model in (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.3-codex-spark",
        "Fable 5",
        "Opus 4.8",
        "Sonnet 5",
    ):
        if required_model not in agent_text:
            raise AssertionError(
                f"AGENT.md delegation policy is missing model mapping: {required_model}"
            )
    if normalized_agent_text(claude) != normalized_agent_text(agent):
        raise AssertionError(
            "CLAUDE.md and AGENT.md must mirror outside the explicit Codex-only section"
        )

    release_branch_policy = {
        contract: (
            "release/v<canonical-semver>",
            "exact green `main` commit",
            "exact settled release tip is the build, freeze, tag, and publication source",
            "temporary branch based on current `main`",
            "making the candidate an ancestor",
            "build head branch to `release/<tag>`",
            "read-only lock",
        ),
        agent: (
            "release/v<canonical-semver>",
            "exact `main` commit whose required checks are green",
            "exact release-branch tip is the candidate source",
            "temporary admission branch cut from current `main`",
            "make the exact candidate tip an ancestor of `main`",
            "head branch is exactly `release/<tag>`",
            "locked read-only reference",
        ),
        claude: (
            "release/v<canonical-semver>",
            "exact `main` commit whose required checks are green",
            "exact release-branch tip is the candidate source",
            "temporary admission branch cut from current `main`",
            "make the exact candidate tip an ancestor of `main`",
            "head branch is exactly `release/<tag>`",
            "locked read-only reference",
        ),
        contributing: (
            "release/v<canonical-semver>",
            "green `main` commit",
            "exact release-branch tip is the build, freeze, tag, and publication source",
            "temporary admission branch from current `main`",
            "current release tip to resolve to the same SHA",
            "exact-ref read-only lock",
        ),
        build_doc: (
            "pull requests, pushes to `main` and protected `release/v*` candidate branches",
            "manual dispatch",
            "candidate lifecycle and eligible frozen-build identity",
        ),
        release_branches: (
            "release/v<canonical-semver>",
            "green `main`",
            "protected release-branch tip as the candidate source",
            "temporary admission branch from the current green `main`",
            "unchanged release tip is now an ancestor of `main`",
            "`head_branch` is exactly `release/$tag`",
            "exact-ref protection",
        ),
    }
    for path, phrases in release_branch_policy.items():
        policy_text = " ".join(path.read_text(encoding="utf-8").split())
        for phrase in phrases:
            if phrase not in policy_text:
                raise AssertionError(
                    f"{path.relative_to(ROOT)} missing release-branch policy: {phrase}"
                )

    cargo_config = tomllib.loads(cargo_manifest.read_text(encoding="utf-8"))
    if cargo_config["workspace"]["package"].get("rust-version") != "1.89":
        raise AssertionError("Cargo.toml must declare the proven Rust 1.89 minimum")
    clippy_policy = tomllib.loads(clippy_config.read_text(encoding="utf-8"))
    if clippy_policy.get("msrv") != "1.89":
        raise AssertionError("Clippy must review against the declared Rust 1.89 MSRV")
    safety_lints = cargo_config["workspace"].get("lints", {}).get("clippy", {})
    for lint in ("missing_safety_doc", "undocumented_unsafe_blocks"):
        if safety_lints.get(lint) != "deny":
            raise AssertionError(f"workspace Clippy policy must deny {lint}")
    for crate_manifest in crate_manifests:
        crate_config = tomllib.loads(crate_manifest.read_text(encoding="utf-8"))
        if crate_config.get("lints", {}).get("workspace") is not True:
            raise AssertionError(
                f"{crate_manifest.relative_to(ROOT)} must inherit workspace lints"
            )

    workflow_text = workflow.read_text(encoding="utf-8")
    release_workflow_text = release_workflow.read_text(encoding="utf-8")
    for protected_workflow in (workflow, release_workflow, native_workflow):
        _validate_protected_branch_triggers(protected_workflow)
    for toolchain in ("1.89.0", "1.97.1"):
        install = (
            f"rustup toolchain install {toolchain} --profile minimal "
            "--component clippy,rustfmt"
        )
        if install not in workflow_text:
            raise AssertionError(
                f"contract CI must install Clippy and rustfmt for Rust {toolchain}"
            )
    for command in (
        "cargo +1.89.0 fmt --all -- --check",
        "cargo +1.89.0 check --workspace --all-targets --locked",
        "cargo +1.89.0 clippy --workspace --all-targets --locked -- -D warnings",
        "cargo +1.89.0 test --workspace --locked --quiet",
    ):
        if command not in workflow_text:
            raise AssertionError(
                f"contract CI must validate the exact Rust 1.89 MSRV: {command}"
            )
    for command in (
        "cargo +1.97.1 fmt --all -- --check",
        "cargo +1.97.1 check --workspace --all-targets --locked",
        "cargo +1.97.1 clippy --workspace --all-targets --locked -- -D warnings",
        "cargo +1.97.1 test --workspace --locked --quiet",
    ):
        if command not in workflow_text:
            raise AssertionError(
                f"contract CI must validate the exact Rust 1.97.1 release compiler: {command}"
            )
    if "RUST_VERSION: '1.97.1'" not in release_workflow_text:
        raise AssertionError("release wheels must pin the exact Rust 1.97.1 compiler")
    rust_evidence_text = rust_release_evidence.read_text(encoding="utf-8")
    for decision in (
        "MSRV:             1.89",
        "release compiler: exact 1.97.1",
        'RUSTFLAGS="-C linker-features=-lld"',
        "312 passed, 13 skipped",
    ):
        if decision not in rust_evidence_text:
            raise AssertionError(
                f"Rust release-compiler evidence is missing: {decision}"
            )
    architecture_gate_text = architecture_gate.read_text(encoding="utf-8")
    if '"--workspace", "--", "--test-threads=1"' not in architecture_gate_text:
        raise AssertionError(
            "the integrated GPU workspace gate must serialize Rust tests"
        )

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if any(line.strip() == "CLAUDE.md" for line in gitignore):
        raise AssertionError(
            "CLAUDE.md must not be ignored; it is tracked project contract, not scratch memory"
        )

    print("contract policy files verified")


if __name__ == "__main__":
    main()
