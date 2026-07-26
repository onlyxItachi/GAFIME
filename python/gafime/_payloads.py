"""Deterministic discovery for installed v1 GPU payload libraries.

The Rust GPU loader intentionally accepts only explicit library paths. This
module fills those paths from trusted, installed GAFIME distributions before
the native boundary resolves a backend. It never replaces a caller-provided
environment value.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import sys


CUDA_LIBRARY_ENV = "GAFIME_CUDA_V1_LIB"
ROCM_LIBRARY_ENV = "GAFIME_ROCM_V1_LIB"
METAL_LIBRARY_ENV = "GAFIME_METAL_V1_LIB"
METAL_METALLIB_ENV = "GAFIME_METAL_V1_METALLIB"

_METAL_LIBRARY_NAME = "libgafime_metal_v1.dylib"
_METAL_METALLIB_NAME = "gafime_metal_v1.metallib"


class PayloadDiscoveryError(RuntimeError):
    """An installed GAFIME payload has an unsafe or incomplete layout."""


@dataclass(frozen=True)
class _PackagePayload:
    backend: str
    distribution: str
    package: str
    env_var: str
    library_names: tuple[str, ...]


_PACKAGE_PAYLOADS = (
    _PackagePayload(
        backend="cuda",
        distribution="gafime-cuda",
        package="gafime_cuda",
        env_var=CUDA_LIBRARY_ENV,
        library_names=(
            "gafime_cuda.dll",
            "libgafime_cuda.so",
            "gafime_cuda.so",
            "gafime_cuda.pyd",
        ),
    ),
    _PackagePayload(
        backend="cuda",
        distribution="gafime-cuda-rt",
        package="gafime_cuda_rt",
        env_var=CUDA_LIBRARY_ENV,
        library_names=(
            "gafime_cuda_rt.dll",
            "libgafime_cuda_rt.so",
            "gafime_cuda_rt.so",
            "gafime_cuda_rt.pyd",
        ),
    ),
    _PackagePayload(
        backend="rocm",
        distribution="gafime-rocm",
        package="gafime_rocm",
        env_var=ROCM_LIBRARY_ENV,
        library_names=(
            "gafime_rocm.dll",
            "libgafime_rocm.so",
            "gafime_rocm.so",
            "gafime_rocm.pyd",
        ),
    ),
    _PackagePayload(
        backend="rocm",
        distribution="gafime-rocm-bundled",
        package="gafime_rocm_bundled",
        env_var=ROCM_LIBRARY_ENV,
        library_names=(
            "libgafime_rocm_bundled.so",
            "gafime_rocm_bundled.so",
            "gafime_rocm_bundled.pyd",
        ),
    ),
    _PackagePayload(
        backend="metal",
        distribution="gafime-metal",
        package="gafime_metal",
        env_var=METAL_LIBRARY_ENV,
        library_names=(_METAL_LIBRARY_NAME,),
    ),
)


def discover_payloads(backend: str | None = None) -> dict[str, Path]:
    """Populate absent GPU library environment variables from installed wheels.

    Discovery is intentionally restricted to the backend that will be resolved,
    or to all platform-supported payloads for ``auto``. Explicit environment
    values, including an explicitly empty one, remain untouched.
    """
    requested = _backends_for_request(backend)
    platform_name, machine = _current_platform()
    discovered: dict[str, Path] = {}

    for backend_name in ("cuda", "rocm"):
        variants = tuple(
            payload for payload in _PACKAGE_PAYLOADS if payload.backend == backend_name
        )
        env_var = variants[0].env_var
        if backend_name not in requested or env_var in os.environ:
            continue
        if not _platform_supports(backend_name, platform_name, machine):
            continue
        library = _discover_package_backend(backend_name, variants)
        if library is None:
            continue
        os.environ[env_var] = str(library)
        discovered[backend_name] = library

    if (
        "metal" in requested
        and METAL_LIBRARY_ENV not in os.environ
        and _platform_supports("metal", platform_name, machine)
    ):
        metal = _discover_metal_payload()
        if metal is not None:
            library, metallib = metal
            os.environ[METAL_LIBRARY_ENV] = str(library)
            os.environ.setdefault(METAL_METALLIB_ENV, str(metallib))
            discovered["metal"] = library

    return discovered


def installed_payload_build_policy(
    backend: str,
) -> tuple[dict[str, object] | None, str]:
    """Read an installed payload's policy without importing or loading it."""

    normalized = str(backend).strip().lower()
    if normalized == "hip":
        normalized = "rocm"
    variants = tuple(
        payload for payload in _PACKAGE_PAYLOADS if payload.backend == normalized
    )
    if not variants:
        return None, f"{normalized} has no separately installed payload policy."

    installations = [
        (payload, distribution, root)
        for payload in variants
        for distribution, root in _matching_distributions(payload.distribution)
    ]
    if not installations:
        return None, f"no installed {normalized} payload distribution was found."

    env_var = variants[0].env_var
    configured_library = os.environ.get(env_var)
    if configured_library is not None:
        configured_path = Path(configured_library).resolve()
        matching_installations = []
        for payload, distribution, root in installations:
            package_dir = _safe_child(root, payload.package, payload.distribution)
            if (
                configured_path.parent == package_dir
                and configured_path.name in payload.library_names
            ):
                matching_installations.append((payload, distribution, root))
        if not matching_installations:
            return (
                None,
                f"{env_var} selects an external library, so no installed-package "
                "policy is attributed to it.",
            )
        installations = matching_installations

    if len(installations) != 1:
        identities = sorted(
            f"{payload.distribution}@{distribution.version}:{root}"
            for payload, distribution, root in installations
        )
        return (
            None,
            f"multiple installed {normalized} payload candidates prevent policy "
            f"attribution: {identities}",
        )

    payload, distribution, root = installations[0]
    expected_version = _core_version()
    if expected_version and distribution.version != expected_version:
        raise PayloadDiscoveryError(
            f"{payload.distribution} version {distribution.version!r} does not match "
            f"gafime version {expected_version!r}. Install matching release artifacts."
        )
    package_dir = _safe_child(root, payload.package, payload.distribution)
    policy_path = _safe_child(package_dir, "build_policy.json", payload.distribution)
    if not policy_path.is_file():
        return (
            None,
            f"installed {payload.distribution} {distribution.version} has no "
            "build_policy.json.",
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PayloadDiscoveryError(
            f"installed {payload.distribution} has an unreadable build policy: {exc}"
        ) from exc
    if not isinstance(policy, dict):
        raise PayloadDiscoveryError(
            f"installed {payload.distribution} build policy must be a JSON object"
        )
    return (
        policy,
        f"installed {payload.distribution} {distribution.version}: {policy_path}",
    )


def _backends_for_request(backend: str | None) -> tuple[str, ...]:
    if backend is None:
        return ("cuda", "rocm", "metal")
    normalized = str(backend).strip().lower()
    if normalized == "hip":
        normalized = "rocm"
    if normalized in {"auto", "gpu"}:
        return ("cuda", "rocm", "metal")
    if normalized in {"cuda", "rocm", "metal"}:
        return (normalized,)
    return ()


def _current_platform() -> tuple[str, str]:
    return sys.platform, platform.machine()


def _platform_supports(
    backend: str,
    platform_name: str | None = None,
    machine: str | None = None,
) -> bool:
    platform_name = (platform_name or sys.platform).lower()
    machine = (machine or platform.machine()).lower().replace("-", "_")
    x86_64 = machine in {"x86_64", "amd64"}
    if backend == "cuda":
        return platform_name in {"linux", "win32"} and x86_64
    if backend == "rocm":
        return platform_name == "linux" and x86_64
    if backend == "metal":
        return platform_name == "darwin" and machine in {"arm64", "aarch64"}
    return False


def _discover_package_backend(
    backend: str, variants: tuple[_PackagePayload, ...]
) -> Path | None:
    installed = [
        (payload, matches)
        for payload in variants
        if (matches := _matching_distributions(payload.distribution))
    ]
    if not installed:
        return None
    if len(installed) != 1:
        distributions = ", ".join(payload.distribution for payload, _ in installed)
        env_var = variants[0].env_var
        raise PayloadDiscoveryError(
            f"multiple installed {backend} payload variants were found: {distributions}. "
            f"Set {env_var} explicitly to the native library to use, or uninstall all but one "
            "variant."
        )
    payload, matches = installed[0]
    return _discover_package_payload(payload, matches)


def _discover_package_payload(
    payload: _PackagePayload,
    matches: list[tuple[metadata.Distribution, Path]] | None = None,
) -> Path | None:
    matches = (
        _matching_distributions(payload.distribution) if matches is None else matches
    )
    if not matches:
        return None
    if len(matches) != 1:
        locations = ", ".join(str(root) for _, root in matches)
        raise PayloadDiscoveryError(
            f"multiple installed {payload.distribution} distributions were found: {locations}. "
            "Keep exactly one payload installation."
        )

    distribution, root = matches[0]
    expected_version = _core_version()
    if expected_version and distribution.version != expected_version:
        raise PayloadDiscoveryError(
            f"{payload.distribution} version {distribution.version!r} does not match "
            f"gafime version {expected_version!r}. Install matching release artifacts."
        )

    package_dir = _safe_child(root, payload.package, payload.distribution)
    if not package_dir.is_dir():
        raise PayloadDiscoveryError(
            f"{payload.distribution} is missing its {payload.package} package directory at "
            f"{package_dir}. Reinstall the payload wheel."
        )

    candidates = [
        _safe_child(package_dir, name, payload.distribution)
        for name in payload.library_names
        if (package_dir / name).exists()
    ]
    if not candidates:
        expected = ", ".join(payload.library_names)
        raise PayloadDiscoveryError(
            f"{payload.distribution} has no native library in {package_dir}; expected one of: "
            f"{expected}. Reinstall the payload wheel."
        )
    invalid = [path for path in candidates if not path.is_file()]
    if invalid:
        raise PayloadDiscoveryError(
            f"{payload.distribution} contains a non-file native payload candidate: "
            f"{', '.join(str(path) for path in invalid)}."
        )
    if len(candidates) != 1:
        raise PayloadDiscoveryError(
            f"{payload.distribution} contains multiple native payload libraries: "
            f"{', '.join(str(path) for path in candidates)}. Keep exactly one."
        )
    return candidates[0]


def _discover_metal_payload() -> tuple[Path, Path] | None:
    variants = tuple(
        payload for payload in _PACKAGE_PAYLOADS if payload.backend == "metal"
    )
    installed_library = _discover_package_backend("metal", variants)
    legacy = _discover_legacy_bundled_metal_payload()
    if installed_library is not None and legacy is not None:
        raise PayloadDiscoveryError(
            "both the gafime-metal distribution and a legacy bundled Metal payload "
            "were found. Install one exact-version Metal payload source."
        )
    if installed_library is not None:
        metallib = _safe_child(
            installed_library.parent, _METAL_METALLIB_NAME, "gafime-metal"
        )
        if not metallib.is_file():
            raise PayloadDiscoveryError(
                "gafime-metal is incomplete; missing paired metallib "
                f"{metallib}. Reinstall the payload wheel."
            )
        return installed_library, metallib
    return legacy


def _discover_legacy_bundled_metal_payload() -> tuple[Path, Path] | None:
    payload_dir = _base_package_dir() / "_metal"
    library = _safe_child(payload_dir, _METAL_LIBRARY_NAME, "gafime Metal")
    metallib = _safe_child(payload_dir, _METAL_METALLIB_NAME, "gafime Metal")
    present = [path.exists() for path in (library, metallib)]
    if not any(present):
        return None
    if not all(present):
        missing = [
            path.name
            for path, is_present in zip((library, metallib), present)
            if not is_present
        ]
        raise PayloadDiscoveryError(
            "the legacy bundled macOS Metal payload is incomplete; missing "
            f"{', '.join(missing)} from {payload_dir}. Reinstall the base wheel."
        )
    invalid = [path for path in (library, metallib) if not path.is_file()]
    if invalid:
        raise PayloadDiscoveryError(
            "the legacy bundled macOS Metal payload contains a non-file artifact: "
            f"{', '.join(str(path) for path in invalid)}."
        )
    return library, metallib


def _matching_distributions(name: str) -> list[tuple[metadata.Distribution, Path]]:
    expected = _canonical_distribution_name(name)
    matches: list[tuple[metadata.Distribution, Path]] = []
    for distribution in metadata.distributions():
        try:
            distribution_name = distribution.metadata.get("Name")
        except Exception:
            continue
        if (
            not distribution_name
            or _canonical_distribution_name(distribution_name) != expected
        ):
            continue
        root = Path(distribution.locate_file("")).resolve()
        matches.append((distribution, root))
    return sorted(matches, key=lambda item: (str(item[1]), item[0].version))


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_version() -> str | None:
    package = sys.modules.get("gafime")
    version = getattr(package, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return metadata.version("gafime")
    except metadata.PackageNotFoundError:
        return None


def _base_package_dir() -> Path:
    return Path(__file__).resolve().parent


def _safe_child(parent: Path, name: str, label: str) -> Path:
    parent = parent.resolve()
    child = (parent / name).resolve()
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise PayloadDiscoveryError(
            f"{label} payload path escapes its package directory: {child}"
        ) from exc
    return child
