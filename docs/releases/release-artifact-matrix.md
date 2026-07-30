# GAFIME Release Artifact Matrix

<!-- Generated from .github/release-artifacts.json; do not edit by hand. -->

The standard GitHub release bundle contains **42 artifacts**, derived from the manifest's per-CPython/platform matrix. Dedicated wheels are built and tested for CPython `3.10`, `3.11`, `3.12`, `3.13`, `3.14`; explicit runner limits are listed below. Python's Stable ABI is not used.

| Distribution | Kind | Runtime policy | Wheel platforms | Embedded backends | Sdist | PyPI publication | Count |
|---|---|---|---|---|---:|---|---:|
| `gafime` | core | Core; Metal embedded on Apple Silicon | `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64`, `win_amd64`, `win_arm64` | `metal` in `macosx_11_0_arm64` | yes | wheels, sdist | 25 |
| `gafime-cuda` | payload | system CUDA runtime | `manylinux_2_28_x86_64`, `win_amd64` | none | yes | wheels, sdist | 11 |
| `gafime-rocm` | payload | system ROCM runtime | `linux_x86_64` | none | yes | sdist | 6 |

## Hosted Validation Limits

- `gafime` / `win_arm64`: hosted runtime validation and wheel production cover `3.11`, `3.12`, `3.13`, `3.14`. actions/python-versions does not publish a native Windows ARM64 CPython 3.10 runtime.
