#!/usr/bin/env python3
"""Report the branch-only #73 probe's actual source cost and isolation.

This is an experiment-specific accounting tool, not a production size gate or
a claim that code-line count measures runtime efficiency. Run after choosing
the exact base revision; no source files or Git refs are changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = "crates/gafime-cpu/examples/issue73_probe/"
GROUPS = {
    "native_implementation_including_demo": [PROBE + "main.rs", PROBE + "probe.rs"],
    "native_tests": [
        PROBE + "tests.rs",
        "crates/gafime-cpu/tests/issue73_evidence_feasibility.rs",
    ],
    "accounting_tool": ["tests/release_measure/issue73_probe_scope.py"],
    "design_record": ["docs/issue-73-native-evidence-feasibility.md"],
}
ALLOWED = {path for paths in GROUPS.values() for path in paths} | {"docs/README.md"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def counts(path: str) -> dict[str, int]:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    comment_prefixes = ("//",) if path.endswith(".rs") else ("#",)
    return {
        "physical_lines": len(lines),
        "nonblank_lines": sum(bool(line.strip()) for line in lines),
        # Deliberately transparent lexical measure, not a Rust/Python parser.
        "nonblank_non_linecomment_lines": sum(
            bool(line.strip()) and not line.lstrip().startswith(comment_prefixes)
            for line in lines
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, help="Exact pre-experiment Git revision"
    )
    args = parser.parse_args()
    base = git("rev-parse", "--verify", f"{args.base}^{{commit}}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT, check=True
    )
    changed = set(git("diff", "--name-only", base, "--").splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    unexpected = changed - ALLOWED
    if unexpected:
        raise SystemExit(f"Changes outside the nonshipping probe: {sorted(unexpected)}")
    files = {path: counts(path) for paths in GROUPS.values() for path in paths}
    totals = {
        group: {
            key: sum(files[path][key] for path in paths)
            for key in next(iter(files.values()))
        }
        for group, paths in GROUPS.items()
    }
    print(
        json.dumps(
            {
                "base_sha": base,
                "head_sha": git("rev-parse", "HEAD"),
                "working_tree_dirty": bool(git("status", "--porcelain")),
                "scope": "nonshipping Core mixed evidence experiment",
                "shipping_runtime_abi_dependency_version_files_unchanged": True,
                "changed_files": sorted(changed),
                "files": files,
                "totals": totals,
                "provisional_600_line_envelope": totals[
                    "native_implementation_including_demo"
                ]["nonblank_non_linecomment_lines"]
                <= 600,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
