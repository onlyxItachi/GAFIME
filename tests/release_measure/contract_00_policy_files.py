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


def normalized_agent_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 3 and lines[2].startswith("This file mirrors `"):
        lines[2] = "This file mirrors `<mirror>`. Keep both files synchronized except for agent-specific notes that are explicitly needed."
    return "\n".join(lines)


def main() -> None:
    contract = ROOT / "docs" / "contract.md"
    claude = ROOT / "CLAUDE.md"
    agent = ROOT / "AGENT.md"
    workflow = ROOT / ".github" / "workflows" / "v1_contract_validation.yml"

    for path in (contract, claude, agent, workflow):
        if not path.exists():
            raise AssertionError(f"required contract artifact is missing: {path.relative_to(ROOT)}")

    contract_text = contract.read_text(encoding="utf-8")
    for section in REQUIRED_CONTRACT_SECTIONS:
        if section not in contract_text:
            raise AssertionError(f"docs/contract.md missing section: {section}")
    for phrase in (
        "CUDA payloads must compile both `kernels.cu` and `launcher.cu`",
        "ROCm payloads must compile both `kernels.hip` and `launcher.hip`",
        "Packaging must not reintroduce top-level GPU source homes",
    ):
        if phrase not in contract_text:
            raise AssertionError(f"docs/contract.md missing GPU packaging rule: {phrase}")

    if normalized_agent_text(claude) != normalized_agent_text(agent):
        raise AssertionError("CLAUDE.md and AGENT.md must mirror each other except the mirror-reference line")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    if any(line.strip() == "CLAUDE.md" for line in gitignore):
        raise AssertionError("CLAUDE.md must not be ignored; it is tracked project contract, not scratch memory")

    print("contract policy files verified")


if __name__ == "__main__":
    main()
