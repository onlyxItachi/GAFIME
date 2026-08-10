#!/usr/bin/env python3
"""Compile and run the standalone Core arithmetic benchmark.

The benchmark source always comes from the clean harness checkout.  The only
product-specific Rust input to the compile is the explicitly named
``gafime-cpu`` rlib (plus its dependency search directory), so a baseline
checkout cannot substitute or mutate the common A/B harness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HARNESS_SOURCE = Path("tests/release_measure/core_precision_native_benchmark.rs")
HARNESS_RUNNER = Path("tests/release_measure/run_core_precision_native_benchmark.py")


@dataclass(frozen=True)
class RepositoryIdentity:
    root: Path
    commit: str
    tree: str
    git_dir: Path
    git_common_dir: Path


@dataclass(frozen=True)
class GitIdentity:
    executable: Path
    sha256: str
    version: str
    trusted_path: str
    sanitized_environment_variables: tuple[str, ...]


@dataclass(frozen=True)
class SourceIdentity:
    path: Path
    relative_path: Path
    sha256: str
    git_blob: str


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


_TRUSTED_GIT_DIRECTORIES = (
    Path("/usr/bin"),
    Path("/bin"),
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
    Path("/opt/local/bin"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_git_executable() -> Path:
    """Resolve Git without trusting an inherited PATH entry.

    The benchmark may be launched with a hostile PATH (for example, a wrapper
    named ``git`` that reports a synthetic commit).  Prefer the conventional
    system locations and only search those same locations as a fallback.  A
    PATH-only Git is deliberately rejected rather than allowing an untrusted
    wrapper to become benchmark provenance.
    """

    candidates: list[Path] = []
    for directory in _TRUSTED_GIT_DIRECTORIES:
        candidates.append(directory / "git")
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    trusted_path = os.pathsep.join(str(path) for path in _TRUSTED_GIT_DIRECTORIES)
    discovered = shutil.which("git", path=trusted_path)
    if discovered:
        resolved = Path(discovered).resolve(strict=True)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise RuntimeError(
        "could not resolve a trusted absolute Git executable from system locations"
    )


def _git_environment(git_executable: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return a Git child environment with inherited GIT_* controls removed."""

    environment = os.environ.copy()
    inherited = tuple(sorted(key for key in environment if key.startswith("GIT_")))
    for key in inherited:
        environment.pop(key, None)
    # Do not allow global/system config to redirect object/config resolution.
    # Local repository config remains available for ordinary Git operation, but
    # all process-level redirection/config/object variables are controlled here.
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "PATH": os.pathsep.join(
                [str(git_executable.parent)]
                + [str(path) for path in _TRUSTED_GIT_DIRECTORIES]
            ),
        }
    )
    return environment, inherited


def _git_identity() -> GitIdentity:
    executable = _resolve_git_executable()
    environment, sanitized = _git_environment(executable)
    version = _run([str(executable), "--version"], env=environment).stdout.strip()
    if not version.startswith("git version "):
        raise RuntimeError("trusted Git executable returned an invalid version")
    return GitIdentity(
        executable=executable,
        sha256=_sha256(executable),
        version=version,
        trusted_path=environment["PATH"],
        sanitized_environment_variables=sanitized,
    )


def _git(root: Path, *arguments: str, git: GitIdentity | None = None) -> str:
    identity = git or _git_identity()
    environment, _ = _git_environment(identity.executable)
    return _run(
        [str(identity.executable), "-C", str(root), *arguments], env=environment
    ).stdout.strip()


def _full_hex(value: str, length: int, label: str) -> str:
    if len(value) != length or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{label} must be a {length}-character hexadecimal identity")
    return value.lower()


def _repository_identity(root: Path) -> RepositoryIdentity:
    resolved = root.resolve(strict=True)
    git = _git_identity()
    reported_top_level = Path(_git(resolved, "rev-parse", "--show-toplevel", git=git))
    if reported_top_level.resolve(strict=True) != resolved:
        raise ValueError(
            "Git reported a repository top-level different from the physical root: "
            f"{reported_top_level} != {resolved}"
        )
    reported_git_dir = Path(_git(resolved, "rev-parse", "--git-dir", git=git))
    if not reported_git_dir.is_absolute():
        reported_git_dir = resolved / reported_git_dir
    reported_git_dir = reported_git_dir.resolve(strict=True)
    dot_git = resolved / ".git"
    if dot_git.is_dir():
        expected_git_dir = dot_git.resolve(strict=True)
    elif dot_git.is_file():
        marker = dot_git.read_text(encoding="utf-8").splitlines()
        if not marker or not marker[0].startswith("gitdir:"):
            raise ValueError(f"invalid linked-worktree .git file: {dot_git}")
        linked = Path(marker[0][len("gitdir:") :].strip())
        if not linked.is_absolute():
            linked = dot_git.parent / linked
        expected_git_dir = linked.resolve(strict=True)
    else:
        raise ValueError(
            f"repository has no .git directory or linked-worktree file: {resolved}"
        )
    if reported_git_dir != expected_git_dir:
        raise ValueError(
            "Git reported a git-dir different from the repository's physical .git "
            f"target: {reported_git_dir} != {expected_git_dir}"
        )
    reported_common_dir = Path(_git(resolved, "rev-parse", "--git-common-dir", git=git))
    if not reported_common_dir.is_absolute():
        reported_common_dir = resolved / reported_common_dir
    reported_common_dir = reported_common_dir.resolve(strict=True)
    if dot_git.is_dir():
        expected_common_dir = expected_git_dir
    elif expected_git_dir.parent.name == "worktrees":
        expected_common_dir = expected_git_dir.parent.parent
    else:
        expected_common_dir = expected_git_dir.parent
    if reported_common_dir != expected_common_dir:
        raise ValueError(
            "Git reported a git-common-dir different from the repository's "
            f"physical common dir: {reported_common_dir} != {expected_common_dir}"
        )
    if _git(resolved, "status", "--porcelain=v1", "--untracked-files=all", git=git):
        raise ValueError(f"repository must be clean: {resolved}")
    return RepositoryIdentity(
        root=resolved,
        commit=_full_hex(_git(resolved, "rev-parse", "HEAD", git=git), 40, "commit"),
        tree=_full_hex(_git(resolved, "rev-parse", "HEAD^{tree}", git=git), 40, "tree"),
        git_dir=reported_git_dir,
        git_common_dir=reported_common_dir,
    )


def _tracked_source_identity(root: Path, relative_path: Path) -> SourceIdentity:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("harness source path must be repository relative")
    source = (root / relative_path).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("harness source escaped its repository") from error
    _git(root, "ls-files", "--error-unmatch", relative_path.as_posix())
    current_blob = _full_hex(
        _git(root, "hash-object", relative_path.as_posix()), 40, "current source blob"
    )
    head_blob = _full_hex(
        _git(root, "rev-parse", f"HEAD:{relative_path.as_posix()}"),
        40,
        "HEAD source blob",
    )
    if current_blob != head_blob:
        raise ValueError("harness source differs from its checked-in HEAD blob")
    return SourceIdentity(
        path=source,
        relative_path=relative_path,
        sha256=_sha256(source),
        git_blob=head_blob,
    )


def _compiler_command(
    *,
    rustup: str,
    toolchain: str,
    source: Path,
    product_rlib: Path,
    binary: Path,
) -> list[str]:
    # The source argument is the common harness path, never a source from the
    # product checkout.  --extern is exact and prevents Cargo target discovery
    # from compiling a product-local benchmark target.
    return [
        rustup,
        "run",
        toolchain,
        "rustc",
        "--crate-name",
        "gafime_core_precision_native_benchmark",
        "--edition=2021",
        str(source),
        "--extern",
        f"gafime_cpu={product_rlib}",
        "-L",
        f"dependency={product_rlib.parent}",
        "-Copt-level=3",
        "-Ccodegen-units=1",
        "-Clto=fat",
        "-Cembed-bitcode=yes",
        "-o",
        str(binary),
    ]


def _compiler_environment(
    source: SourceIdentity,
    runner_source: SourceIdentity,
    product_rlib: Path,
    compiler_command: list[str],
) -> dict[str, str]:
    environment, _ = _git_environment(_git_identity().executable)
    command_json = json.dumps(
        compiler_command, ensure_ascii=True, separators=(",", ":")
    )
    environment.update(
        {
            "GAFIME_COMPILED_HARNESS_SOURCE_SHA256": source.sha256,
            "GAFIME_COMPILED_HARNESS_SOURCE_GIT_BLOB": source.git_blob,
            "GAFIME_COMPILED_HARNESS_SOURCE_RELATIVE_PATH": source.relative_path.as_posix(),
            "GAFIME_COMPILED_HARNESS_RUNNER_SHA256": runner_source.sha256,
            "GAFIME_COMPILED_HARNESS_RUNNER_GIT_BLOB": runner_source.git_blob,
            "GAFIME_COMPILED_HARNESS_RUNNER_RELATIVE_PATH": runner_source.relative_path.as_posix(),
            "GAFIME_COMPILED_PRODUCT_RLIB_SHA256": _sha256(product_rlib),
            # Embed a fixed-width identity rather than the variable-length
            # command itself. Output and dependency paths must not shift hot
            # function alignment in the binary under measurement.
            "GAFIME_COMPILED_COMMAND_SHA256": hashlib.sha256(
                command_json.encode("utf-8")
            ).hexdigest(),
        }
    )
    return environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-source-root", type=Path, required=True)
    parser.add_argument("--harness-source-root", type=Path, required=True)
    parser.add_argument("--product-rlib", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-policy", choices=("common-f64", "native"), required=True
    )
    parser.add_argument("--seed", type=int, default=0x51A7_2026_0809)
    parser.add_argument("--toolchain", default="1.97.1")
    parser.add_argument("--rustup", default="rustup")
    parser.add_argument("--expected-product-commit")
    parser.add_argument("--expected-harness-commit")
    return parser


def _one_line(command: list[str]) -> str:
    completed = _run(command)
    return (completed.stdout or completed.stderr).splitlines()[0].strip()


def _augment_report(
    output: Path,
    *,
    git: GitIdentity,
    repositories: tuple[RepositoryIdentity, ...],
) -> None:
    """Attach the authenticated Git tool and root checks to child evidence."""

    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("standalone benchmark output is not valid JSON") from error
    git_record = {
        "path": str(git.executable),
        "executable": str(git.executable),
        "sha256": git.sha256,
        "version": git.version,
        "trusted_path": git.trusted_path,
        "removed_environment": list(git.sanitized_environment_variables),
        "sanitized_environment_variables": list(git.sanitized_environment_variables),
        "controlled_environment_variables": [
            "GIT_CONFIG_NOSYSTEM",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
        ],
        "path_lookup_ignored": True,
    }
    report["git_provenance"] = {
        **git_record,
        "repositories": [
            {
                "root": str(repository.root),
                "commit": repository.commit,
                "tree": repository.tree,
                "git_dir": str(repository.git_dir),
                "git_common_dir": str(repository.git_common_dir),
                "show_toplevel_verified": True,
                "git_dir_verified": True,
                "clean_tree_verified": True,
            }
            for repository in repositories
        ],
    }
    report["git"] = git_record
    report["git_identity"] = git_record
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        provenance["git"] = git_record
    output.write_text(
        json.dumps(report, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.seed < 0 or args.seed > (1 << 64) - 1:
        raise ValueError("--seed must fit an unsigned 64-bit integer")
    git_identity = _git_identity()
    product = _repository_identity(args.product_source_root)
    harness = _repository_identity(args.harness_source_root)
    if (
        args.expected_product_commit
        and product.commit != args.expected_product_commit.lower()
    ):
        raise ValueError("product checkout does not match --expected-product-commit")
    if (
        args.expected_harness_commit
        and harness.commit != args.expected_harness_commit.lower()
    ):
        raise ValueError("harness checkout does not match --expected-harness-commit")
    source = _tracked_source_identity(harness.root, HARNESS_SOURCE)
    runner_source = _tracked_source_identity(harness.root, HARNESS_RUNNER)
    product_rlib = args.product_rlib.resolve(strict=True)
    wheel = args.wheel.resolve(strict=True)
    if not product_rlib.is_file() or product_rlib.suffix != ".rlib":
        raise ValueError("--product-rlib must name the exact gafime-cpu .rlib")
    if "gafime_cpu" not in product_rlib.name:
        raise ValueError("--product-rlib filename must identify gafime_cpu")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("--wheel must name the exact Core wheel")

    binary = args.binary.resolve()
    output = args.output.resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = _compiler_command(
        rustup=args.rustup,
        toolchain=args.toolchain,
        source=source.path,
        product_rlib=product_rlib,
        binary=binary,
    )
    compile_environment = _compiler_environment(
        source, runner_source, product_rlib, compiler
    )
    compiler_command_json = json.dumps(
        compiler, ensure_ascii=True, separators=(",", ":")
    )
    _run(compiler, env=compile_environment)
    rustc_version = _one_line(
        [args.rustup, "run", args.toolchain, "rustc", "--version"]
    )
    try:
        linker_version = _one_line(["cc", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        linker_version = _one_line(["ld", "--version"])

    environment, _ = _git_environment(git_identity.executable)
    environment.update(
        {
            "GAFIME_NATIVE_PRODUCT_SOURCE_ROOT": str(product.root),
            "GAFIME_NATIVE_HARNESS_SOURCE_ROOT": str(harness.root),
            "GAFIME_NATIVE_HARNESS_SOURCE": str(source.path),
            "GAFIME_NATIVE_HARNESS_SOURCE_SHA256": source.sha256,
            "GAFIME_NATIVE_HARNESS_SOURCE_GIT_BLOB": source.git_blob,
            "GAFIME_NATIVE_HARNESS_RUNNER": str(runner_source.path),
            "GAFIME_NATIVE_HARNESS_RUNNER_SHA256": runner_source.sha256,
            "GAFIME_NATIVE_HARNESS_RUNNER_GIT_BLOB": runner_source.git_blob,
            "GAFIME_NATIVE_EXPECTED_PRODUCT_COMMIT": product.commit,
            "GAFIME_NATIVE_EXPECTED_PRODUCT_TREE": product.tree,
            "GAFIME_NATIVE_EXPECTED_HARNESS_COMMIT": harness.commit,
            "GAFIME_NATIVE_EXPECTED_HARNESS_TREE": harness.tree,
            "GAFIME_NATIVE_PRODUCT_RLIB": str(product_rlib),
            "GAFIME_NATIVE_PRODUCT_RLIB_SHA256": _sha256(product_rlib),
            "GAFIME_NATIVE_BENCH_WHEEL": str(wheel),
            "GAFIME_NATIVE_BENCH_WHEEL_SHA256": _sha256(wheel),
            "GAFIME_NATIVE_BENCH_BINARY_SHA256": _sha256(binary),
            "GAFIME_NATIVE_BENCH_OUTPUT": str(output),
            "GAFIME_NATIVE_INPUT_POLICY": args.input_policy,
            "GAFIME_NATIVE_BENCH_SEED": str(args.seed),
            "GAFIME_NATIVE_RUSTC_VERSION": rustc_version,
            "GAFIME_NATIVE_LINKER_VERSION": linker_version,
            "GAFIME_NATIVE_COMPILER_COMMAND_JSON": compiler_command_json,
            "GAFIME_NATIVE_GIT_EXECUTABLE": str(git_identity.executable),
            "GAFIME_NATIVE_GIT_SHA256": git_identity.sha256,
            "GAFIME_NATIVE_GIT_VERSION": git_identity.version,
            "GAFIME_NATIVE_GIT_TRUSTED_PATH": git_identity.trusted_path,
            "GAFIME_NATIVE_GIT_SANITIZED_VARIABLES": ",".join(
                git_identity.sanitized_environment_variables
            ),
        }
    )
    completed = _run([str(binary)], env=environment, capture_output=False)
    if completed.returncode != 0:
        return completed.returncode
    if not output.is_file():
        raise RuntimeError("standalone benchmark did not produce --output")
    _augment_report(
        output,
        git=git_identity,
        repositories=(product, harness),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Core native benchmark runner failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
