# ROCm Distribution Policy

GAFIME implements two explicit, immutable ROCm packaging policies. They use
different distribution identities and cannot be substituted after staging:

| Policy | Distribution | Userspace | Standard release |
|---|---|---|---|
| `system` | `gafime-rocm` | Host-provided ROCm 7.2.x | Yes |
| `bundled` | `gafime-rocm-bundled` | Wheel-private pinned closure | No |

Unset, unknown, or conflicting policy requests fail before compilation. The
machine-readable contracts are
`.github/scripts/rocm_7_2_3_system_policy.json` and
`.github/scripts/rocm_7_2_3_bundled_policy.json`.

## Standard System Policy

The `v1.0.0b1` standard payload is compiled with pinned ROCm 7.2.3 build inputs
for the contracted 13-target GFX set. Its wheel contains the GAFIME payload
only. It must:

- contain no `libamd*`, HIP, HSA, or other ROCm userspace libraries;
- contain no wheel-private ROCm library directory, SBOM, RPATH, or RUNPATH;
- depend directly on exactly one ROCm runtime SONAME,
  `libamdhip64.so.7`;
- remain below the checked-in compressed, uncompressed, and native-payload
  size ceilings;
- report `wheel_policy="system"` and `userspace_bundled=false` through public
  capability diagnostics.

The host must provide one coherent ROCm 7.2.x userspace through its system
dynamic loader. Supported hardware, operating-system, kernel driver, and user
permissions remain AMD prerequisites. GAFIME does not install or update those
components.

The thin wheel retains the truthful `linux_x86_64` platform tag. It is not
retagged as manylinux: `libamdhip64.so.7` is not part of the manylinux external
library contract. PyPI rejects raw Linux wheels, so normal publication:

- attaches the thin wheel to the signed GitHub Release;
- publishes the matching `gafime-rocm` source distribution to PyPI;
- never uploads the thin wheel to PyPI and never manually gives it a false
  manylinux tag.

Installing the PyPI sdist requires a compatible ROCm development toolchain,
including `hipcc`. Users who already have only the runtime prerequisite can
install the matching wheel from the GitHub Release.

## Bundled Policy

The prior repair behavior remains reproducible under the separate
`gafime-rocm-bundled` identity. It uses `auditwheel` to vendor the pinned ROCm
userspace closure and emits component/license metadata, a CycloneDX SBOM, size
reports, relative RPATHs, rewritten SONAMEs, and a closed ELF dependency graph.

Bundled mode is not part of the b1 standard bundle or any PyPI publishing lane.
This avoids silently freezing about 73 MiB of ROCm userspace under the standard
package name and avoids loading a private runtime ahead of system security
updates. It remains available for explicit compatibility investigation and
must pass its own stricter closure gate when built.

## Compatibility Boundary

Both policies require supported AMD hardware and a compatible amdgpu kernel
driver. AMD's
[ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html)
and
[user/kernel-space compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/user-kernel-space-compat-matrix.html)
remain authoritative.

System mode intentionally lets the host own one ROCm userspace. It does not
claim that arbitrary ROCm generations are interchangeable, and it does not
support loading multiple HIP/HSA runtime generations in one process.

## Artifact Gates

The standard wheel gate checks the actual archive and ELF:

- embedded policy equals the checked-in system manifest;
- platform tag is exactly `linux_x86_64`;
- no bundled userspace, repair SBOM, RPATH, or RUNPATH exists;
- direct dependencies match the declared allowlist and runtime SONAME;
- size ceilings hold;
- a clean environment with pinned ROCm 7.2.3 resolves and loads the payload.

The wheel is tested through the same CPython 3.10 Stable ABI artifact on Python
3.10, 3.11, 3.12, 3.13, and 3.14. The workflow uploads
`rocm-wheel-policy-report.json` as deterministic packaging evidence. This is not
a performance claim or a legal opinion.

## Build And Diagnostics

Stage the standard system payload:

```bash
python .github/scripts/stage_gpu_payload.py rocm payload-src/gafime-rocm \
  --rocm-wheel-policy system
```

Stage the separately identified bundled payload:

```bash
python .github/scripts/stage_gpu_payload.py rocm \
  payload-src/gafime-rocm-bundled \
  --rocm-wheel-policy bundled
```

Inspect an installed package without running a scoring workload:

```bash
gafime --check --backend rocm
```

The capability source is `package` only when the policy came from one uniquely
matching installed distribution. An explicit external library path is reported
as unknown rather than attributed to an unrelated wheel.

Published artifacts are immutable. This policy changes only `v1.0.0b1` and
later artifacts; it does not modify `v1.0.0b0`.
