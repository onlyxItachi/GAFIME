# ROCm Wheel Policy

GAFIME has one reviewed ROCm wheel policy: `bundled`. It is selected explicitly
while staging `gafime-rocm` and becomes immutable in the staged source, source
distribution, and wheel. `system`, `amd-wheels`, and unknown policy names fail
before compilation.

The machine-readable source of truth is
`.github/scripts/rocm_7_2_3_bundled_policy.json`. The manifest records the ROCm
version, code-object targets, manylinux baseline, component package/version and
license identities, driver compatibility source, size limits, and runtime
coexistence policy.

## Bundled Boundary

The Linux `manylinux_2_28_x86_64` wheel is built against the pinned ROCm 7.2.3
EL8 repository and repaired by `auditwheel`. Repair places the required ROCm
userspace dependency closure in the wheel next to the GAFIME payload. It does
not bundle an AMD kernel driver, firmware, GPU device, or operating-system
support.

The build policy pins the manylinux image by digest, the ROCm signing-key
SHA-256, and the exact HIP compiler/device-library and C++ development package
versions. A repository or package update therefore fails the build or the
artifact manifest instead of silently changing the wheel closure.

This policy exists so an installed `gafime-rocm` wheel does not depend on an
unversioned `/opt/rocm` userspace tree. The host must still provide supported
AMD hardware, operating system, and amdgpu driver. AMD's
[ROCm 7.2.3 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.2.3/compatibility/compatibility-matrix.html)
and
[user/kernel-space compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/user-kernel-space-compat-matrix.html)
remain authoritative. The checked-in policy records the documented compatible
amdgpu series `6.4.x`, `30.10.x`, `30.20.x`, and `30.30.x`; GPU- and OS-specific
restrictions still apply.

Loading this bundled userspace in the same process as another ROCm userspace is
unsupported. Hashed private library names reduce accidental collisions, but
they do not establish that two HIP/HSA runtime generations can coexist safely.

## Artifact Gate

Every candidate ROCm wheel must pass all of these checks:

- the embedded build policy exactly matches the checked-in manifest;
- compressed wheel, uncompressed wheel, native payload, and per-component
  sizes stay within reviewed limits;
- every private shared library maps to exactly one pinned package, version, and
  non-empty license declaration;
- the auditwheel CycloneDX SBOM identifies every bundled package and includes it
  in the wheel's root dependency closure;
- `readelf` confirms private SONAMEs, relative RPATHs, and a closed dependency
  graph; only the explicit manylinux system-library allowlist may resolve
  outside the wheel;
- a clean installed-wheel smoke runs `ldd`, rejects missing or `/opt/rocm`
  dependencies, loads the payload with `RTLD_LOCAL`, and verifies public policy
  diagnostics.

The workflow uploads `rocm-wheel-policy-report.json` with the wheel. The report
contains deterministic component sizes, the policy hash, SBOM path, and the
remaining approved system dependencies. This is packaging evidence, not a
performance claim or a legal opinion.

## Deferred Policies

`system` is not implemented because a wheel that relies on arbitrary host ROCm
userspace would not have a coherent version or dependency contract.
`amd-wheels` is also deferred: the ROCm 7.2.3 package set used by this release
does not provide a reviewed drop-in modular runtime contract for this payload's
native dependency closure. Either mode needs a separate distribution design,
clean-environment proof, and compatibility review before it can be enabled.

## Build And Diagnostics

Stage a local ROCm payload explicitly:

```bash
python .github/scripts/stage_gpu_payload.py rocm payload-src/gafime-rocm \
  --rocm-wheel-policy bundled
```

Inspect an installed package without running a scoring workload:

```bash
gafime --check --backend rocm
```

The capability source is `package` when the policy came from a uniquely
matching installed distribution. An explicit external library path is reported
as unknown rather than being attributed to an unrelated installed wheel.

Published artifacts are immutable. Policy hardening merged after a release
applies to a later version; it cannot change an existing PyPI wheel in place.
