# GAFIME Release Operations

This runbook covers validation and recovery for the split GAFIME distribution.
It does not authorize a tag, PyPI upload, or GitHub Release. A maintainer must
explicitly approve publication from the final reviewed commit.

## Distribution Set

The authoritative distribution, ABI, platform, publication, and artifact-count
contract is [`.github/release-artifacts.json`](../../.github/release-artifacts.json).
Its human-readable generated view is the
[release artifact matrix](release-artifact-matrix.md). Edit the manifest first;
the release gate rejects workflow, optional-extra, or generated-document drift.

The same `cp310-abi3` wheel is tested on CPython 3.10 through 3.14 wherever a
native hosted interpreter exists. Windows ARM64 is explicitly limited to
3.11 through 3.14 because actions/python-versions does not publish a native
Windows ARM64 3.10 runtime; Windows x86_64 still covers 3.10. The artifact is
one Stable ABI wheel, not a Python-3.10-only wheel.

The thin ROCm wheel is attached to the GitHub Release only. PyPI rejects raw
`linux_x86_64` wheels, and its external `libamdhip64.so.7` prerequisite means
it cannot truthfully be retagged as manylinux. The ROCm PyPI lane therefore
publishes only the matching source distribution.

The optional Linux `gafime-cuda-rt` wheel and source distribution are a separate
non-PyPI bundle. They never enter the standard bundle or a PyPI publishing job.

## Publication-Disabled Validation

Run all workflows from the exact candidate commit or branch. Every publication
input must remain false:

```bash
gh workflow run v1_contract_validation.yml --ref <candidate-ref>
gh workflow run native_platform_validation.yml --ref <candidate-ref>
gh workflow run build_wheels.yml --ref <candidate-ref> \
  -f publish_pypi_core=false \
  -f publish_pypi_cuda=false \
  -f publish_pypi_rocm=false \
  -f publish_github_release=false \
  -f build_cuda_rt_payload=false \
  -f allow_matching_existing_pypi_files=false \
  -f check_pypi_collisions=true
```

The wheel run must build and validate the frozen standard bundle declared by
the manifest, then check its filenames against live PyPI. Publishing jobs
should be skipped. A skipped optional RT lane is expected when
`build_cuda_rt_payload=false`.

The ROCm artifact must include `rocm-wheel-policy-report.json`. Review its
policy hash, size totals, `userspace_bundled=false`, platform tag, and required
SONAME against [the ROCm distribution policy](../rocm-wheel-policy.md). The
archive and ELF gates must find no wheel-private ROCm userspace, SBOM, RPATH, or
RUNPATH. The installed smoke must resolve `libamdhip64.so.7` from the pinned
ROCm 7.2.3 system-runtime container. The build log must also show the
digest-pinned build image, signing-key SHA-256 check, and exact package versions
from the policy manifest.

The macOS lane must install the exact local Core wheel and execute its bundled
Metal public API on Apple hardware for every selected Python version. No
separate Metal distribution may be built or installed.

Inspect every job, including expected skips:

```bash
gh run list --branch <candidate-ref> --limit 20
gh run watch <run-id> --exit-status
gh run view <run-id> --json conclusion,jobs,url
```

Do not treat compilation on one architecture as runtime evidence for another.
The release notes must keep compile coverage and runtime-tested hardware
separate.

## Normal Publication

Before creating a version tag:

1. Confirm `pyproject.toml`, Cargo package versions, payload package versions,
   and `docs/releases/v<version>.md` agree.
2. Confirm the candidate commit is on `main` and all required hosted checks pass.
3. Confirm the publication-disabled wheel run produced exactly the expected
   bundle and reported no PyPI collision. Confirm the ROCm policy report and
   installed closure smoke passed in that same run.
4. Confirm the release note no longer describes the version as unissued.
5. Obtain explicit maintainer authorization for the tag and publication.

A push of `v<version>` starts the normal immutable publication chain. The
workflow publishes CUDA and the ROCm sdist first, Core second, and the GitHub
Release last. This ordering prevents a new Core extra from resolving before its
matching external vendor project exists. Metal is already part of the macOS
Core wheel and has no publication lane.

Normal publication fails on any existing PyPI filename. Do not enable recovery
inputs preemptively, and do not use `core_wheel_build_tag` for a normal release.

## Partial-Publication Recovery

A failed tag-push run is intentionally not rerunnable through blind
`skip-existing` behavior. First freeze the original artifacts, inspect which
files reached PyPI, and compare every existing remote file with the local
SHA-256.

For a dispatch recovery from the existing version tag:

- enable only the missing PyPI lanes when no already-published lane must rerun;
- set `allow_matching_existing_pypi_files=true` only when an enabled lane has an
  existing filename and the collision preflight proves the remote and local
  SHA-256 values are identical;
- never skip a mismatched filename;
- use `core_wheel_build_tag` only for the documented blocked/deleted Core-wheel
  recovery case, after review of compatibility and filename consequences.

Example: CUDA is already published, while ROCm and Core must complete:

```bash
gh workflow run build_wheels.yml --ref v<version> \
  -f publish_pypi_core=true \
  -f publish_pypi_cuda=false \
  -f publish_pypi_rocm=true \
  -f publish_github_release=false \
  -f allow_matching_existing_pypi_files=false \
  -f check_pypi_collisions=true
```

If all PyPI files exist and only GitHub Release creation failed, the recovery
dispatch must enable all three PyPI inputs plus GitHub Release and hash-matched
skipping. The three PyPI jobs revalidate their frozen files and report success,
which satisfies the release dependency chain:

```bash
gh workflow run build_wheels.yml --ref v<version> \
  -f publish_pypi_core=true \
  -f publish_pypi_cuda=true \
  -f publish_pypi_rocm=true \
  -f publish_github_release=true \
  -f allow_matching_existing_pypi_files=true \
  -f check_pypi_collisions=true
```

Record the failed run, recovery run, artifact checksums, and exact reason for
recovery in the release handoff. Never rebuild a supposedly identical artifact
and assume its bytes match the frozen publication set.
