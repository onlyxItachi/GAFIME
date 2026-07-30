# ROCm Distribution Policy

GAFIME has one ROCm distribution identity and one runtime policy:

| Distribution | Platform | Runtime ownership | PyPI |
|---|---|---|---|
| `gafime-rocm` | Linux x86_64 | compatible system ROCm 7.2.x | sdist only |

There is no bundled-runtime ROCm distribution. GAFIME wheels and sdists carry
only GAFIME source or binaries; they do not vendor HIP, HSA, COMGR, or other
ROCm userspace components.

## System Policy

The machine-readable policy is
`.github/scripts/rocm_7_2_3_system_policy.json`. The release build uses pinned
ROCm 7.2.3 inputs for the contracted 13-target GFX set. Every generated wheel
must:

- contain only the GAFIME ROCm native payload and package metadata;
- contain no wheel-private ROCm library directory or repair SBOM;
- contain no RPATH or RUNPATH;
- depend directly on exactly one ROCm runtime SONAME,
  `libamdhip64.so.7`;
- retain the truthful `linux_x86_64` platform tag;
- remain below the checked compressed, uncompressed, and native-payload size
  ceilings;
- report `wheel_policy="system"` and `userspace_bundled=false`.

The host owns the kernel driver and one coherent ROCm userspace. GAFIME does
not install, update, or select between host ROCm generations.

## Publication

Raw `linux_x86_64` wheels are not accepted by PyPI, and
`libamdhip64.so.7` is not a manylinux-allowed external dependency. GAFIME does
not repair or falsely retag these wheels.

For each declared CPython version:

- the frozen GitHub Release includes the matching
  `cpXY-cpXY-linux_x86_64` wheel;
- PyPI receives the matching `gafime-rocm` source distribution;
- a public-install gate builds that sdist against the pinned system ROCm
  environment before the GitHub Release is created.

Every payload requires the exact matching `gafime` Core version. Core has no
ROCm dependency or extra.

## Build And Validation

Stage the source payload:

```bash
python .github/scripts/stage_gpu_payload.py rocm payload-src/gafime-rocm \
  --rocm-wheel-policy system
```

Build it with an installed ROCm development toolchain:

```bash
GAFIME_ROCM_ARCHS=<rocm-offload-target> \
  python -m build --wheel payload-src/gafime-rocm
```

Inspect an installed payload:

```bash
gafime --check --backend rocm
```

The release archive gate checks:

- exact policy equality with the checked-in system manifest;
- per-CPython filename and internal wheel tags;
- absence of ROCm userspace and private search paths;
- direct ELF dependencies and the required runtime SONAME;
- artifact size ceilings;
- clean installed discovery and ABI exports;
- public source installation against pinned ROCm 7.2.3.

`rocm-wheel-policy-report.json` is uploaded as build evidence outside the
frozen release bundle. Its schema-v2 `wheels` list covers every
manifest-declared CPython artifact while sharing one checked policy identity.
It is not a performance claim or a legal opinion.

## Compatibility Boundary

Supported hardware, operating system, kernel driver, and permissions remain
AMD prerequisites. AMD's
[ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html)
and
[user/kernel-space compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/user-kernel-space-compat-matrix.html)
remain authoritative.

This policy does not claim that arbitrary ROCm generations are
interchangeable, and it does not support multiple HIP/HSA runtime generations
in one process.
