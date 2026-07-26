# GAFIME Release Artifact Matrix

<!-- Generated from .github/release-artifacts.json; do not edit by hand. -->

The standard GitHub release bundle contains **13 artifacts**. Every wheel is built once with `cp310-abi3` and the same frozen wheel is installed and tested on CPython `3.10`, `3.11`, `3.12`, `3.13`, `3.14`.

| Distribution | Kind | Wheel platforms | Sdist | PyPI publication | Count |
|---|---|---|---:|---|---:|
| `gafime` | core | `manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64`, `win_amd64`, `win_arm64` | yes | wheels, sdist | 6 |
| `gafime-cuda` | payload | `manylinux_2_28_x86_64`, `win_amd64` | yes | wheels, sdist | 3 |
| `gafime-rocm` | payload | `linux_x86_64` | yes | sdist | 2 |
| `gafime-metal` | payload | `macosx_11_0_arm64` | yes | wheels, sdist | 2 |

## Excluded Identities

- `gafime-cuda-rt` (`rt-on`): OptiX-enabled artifacts are separately selected and never enter the standard bundle.
- `gafime-rocm-bundled` (`bundled`): The optional bundled-userspace identity is not promoted until its mixed-runtime contract is verified.
