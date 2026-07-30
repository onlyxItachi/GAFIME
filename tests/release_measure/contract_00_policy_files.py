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
        lines[2] = "This file mirrors `<mirror>`. Keep both files synchronized except for agent-specific notes that are explicitly needed."
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


def main() -> None:
    contract = ROOT / "docs" / "contract.md"
    claude = ROOT / "CLAUDE.md"
    agent = ROOT / "AGENT.md"
    workflow = ROOT / ".github" / "workflows" / "v1_contract_validation.yml"
    cargo_manifest = ROOT / "Cargo.toml"
    architecture_gate = ROOT / "tests" / "release_measure" / "v1_architecture_gate.py"
    release_version = ROOT / ".github" / "scripts" / "release_version.py"
    crate_manifests = tuple(sorted((ROOT / "crates").glob("*/Cargo.toml")))
    gitignore = ROOT / ".gitignore"

    for path in (
        contract,
        claude,
        agent,
        gitignore,
        workflow,
        cargo_manifest,
        architecture_gate,
        release_version,
        *crate_manifests,
    ):
        if not path.exists():
            raise AssertionError(f"required contract artifact is missing: {path.relative_to(ROOT)}")

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
        "Standard CUDA payloads compile only `kernels.cu` and `launcher.cu`",
        "standard ROCm payloads compile both `kernels.hip` and `launcher.hip`",
        "GPU payload staging and release packaging must source backend files from this root `src/` layout",
        "Packaging must not reintroduce `gpu/`, crate-local native source homes",
        "CPU fixed-bin mutual information is the CPU parity path for the GPU-compatible MI approximation",
        "bundle no ROCm userspace, carry no RPATH or RUNPATH",
        "PyPI receives the matching source distribution",
        "There is no bundled-runtime\nROCm distribution policy",
        "There is no RT distribution identity",
        "dynamically requires\nthe system CUDA runtime",
        "must not vendor `libcudart` or `cudart64`",
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
            raise AssertionError(f"docs/contract.md missing GPU packaging rule: {phrase}")

    agent_text = agent.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8")
    if AGENT_ONLY_SECTION not in agent_text or AGENT_ONLY_SECTION in claude_text:
        raise AssertionError(
            "Codex delegation rules must exist only in AGENT.md"
        )
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

    cargo_config = tomllib.loads(cargo_manifest.read_text(encoding="utf-8"))
    if cargo_config["workspace"]["package"].get("rust-version") != "1.89":
        raise AssertionError("Cargo.toml must declare the proven Rust 1.89 minimum")
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
    if "cargo +1.89.0 check --workspace" not in workflow_text:
        raise AssertionError("contract CI must compile the workspace with Rust 1.89")
    if (
        "cargo +1.89.0 clippy --workspace --all-targets --locked -- -D warnings"
        not in workflow_text
    ):
        raise AssertionError(
            "contract CI must keep every workspace target free of Clippy warnings"
        )
    architecture_gate_text = architecture_gate.read_text(encoding="utf-8")
    if '"--workspace", "--", "--test-threads=1"' not in architecture_gate_text:
        raise AssertionError(
            "the integrated GPU workspace gate must serialize Rust tests"
        )

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if any(line.strip() == "CLAUDE.md" for line in gitignore):
        raise AssertionError("CLAUDE.md must not be ignored; it is tracked project contract, not scratch memory")

    print("contract policy files verified")


if __name__ == "__main__":
    main()
