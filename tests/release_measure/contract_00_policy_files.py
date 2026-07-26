#!/usr/bin/env python3
from __future__ import annotations

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
    "## PR, Main, And Release Gates",
    "## Regression Policy",
    "## Migration Rules",
)
AGENT_ONLY_SECTION = "## Delegated Agent Coordination"


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

    for path in (contract, claude, agent, workflow, architecture_gate):
        if not path.exists():
            raise AssertionError(f"required contract artifact is missing: {path.relative_to(ROOT)}")

    contract_text = contract.read_text(encoding="utf-8")
    for section in REQUIRED_CONTRACT_SECTIONS:
        if section not in contract_text:
            raise AssertionError(f"docs/contract.md missing section: {section}")
    for phrase in (
        "CUDA payloads must compile both `kernels.cu` and `launcher.cu`",
        "ROCm payloads must compile both `kernels.hip` and `launcher.hip`",
        "GPU payload staging and release packaging must source backend files from this root `src/` layout",
        "Packaging must not reintroduce `gpu/`, crate-local native source homes",
        "CPU fixed-bin mutual information is the CPU parity path for the GPU-compatible MI approximation",
        "The standard `gafime-rocm` identity uses `system`",
        "bundle no ROCm userspace, carry no RPATH or RUNPATH",
        "PyPI receives the matching source distribution",
        "The separately identified\n`gafime-rocm-bundled` policy",
        "CycloneDX SBOM, size, relative-RPATH, SONAME, and ELF closure",
        "`--scope rocm-bundled-wheel`",
        "`--backend rocm-bundled`",
        "Apple Silicon Metal is the distinct `gafime-metal` distribution",
        "workflow must test that same wheel on CPython 3.10, 3.11, 3.12, 3.13, and 3.14",
    ):
        if phrase not in contract_text:
            raise AssertionError(f"docs/contract.md missing GPU packaging rule: {phrase}")

    agent_text = agent.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8")
    if AGENT_ONLY_SECTION not in agent_text or AGENT_ONLY_SECTION in claude_text:
        raise AssertionError(
            "Codex delegation rules must exist only in AGENT.md"
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

    if 'rust-version = "1.89"' not in cargo_manifest.read_text(encoding="utf-8"):
        raise AssertionError("Cargo.toml must declare the proven Rust 1.89 minimum")
    if "cargo +1.89.0 check --workspace" not in workflow.read_text(encoding="utf-8"):
        raise AssertionError("contract CI must compile the workspace with Rust 1.89")
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
