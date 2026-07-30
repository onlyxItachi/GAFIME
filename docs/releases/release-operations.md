# GAFIME Release Operations

This runbook covers validation, publication, and recovery for the split GAFIME
distribution. It does not authorize a tag, PyPI upload, or GitHub Release.
Publication still requires explicit maintainer approval.

## Pinned Distribution Set

The authoritative per-CPython, platform, publication, and artifact-count
contract is [`.github/release-artifacts.json`](../../.github/release-artifacts.json).
Its generated human-readable view is the
[release artifact matrix](release-artifact-matrix.md).

Permanent rules:

- Python's Stable ABI and `abi3` are not used.
- Core has no CUDA or ROCm dependency or extra.
- CUDA and ROCm payloads depend on the exact matching Core version.
- CUDA wheels target Linux x86_64 and Windows x86_64, contain only GAFIME
  binaries, and require the system CUDA runtime.
- ROCm wheels target Linux x86_64, contain no ROCm userspace, and require the
  system runtime.
- Apple Silicon Metal is embedded in the macOS arm64 Core wheel.
- RT/OptiX is locally buildable through CMake only and never enters release
  artifacts or workflow caches.
- Artifact counts come from the manifest's CPython/platform matrix and are not
  copied into workflow logic.

Windows ARM64 currently builds and validates Python 3.11 through 3.14 because
actions/python-versions has no native Windows ARM64 CPython 3.10 runtime. Every
other declared platform covers Python 3.10 through 3.14. CPython 3.10 users on
Windows ARM64 must move to Python 3.11 through 3.14 or build the sdist locally
with Rust and the MSVC ARM64 toolchain.

The truthful raw ROCm wheels are attached to the GitHub Release. PyPI receives
the matching ROCm sdist because raw `linux_x86_64` wheels are rejected and
`libamdhip64.so.7` cannot truthfully satisfy manylinux.

## Version Identity

Cargo and repository surfaces use SemVer. Python and PyPI use the strict PEP
440 mapping:

```bash
python .github/scripts/release_version.py --check-project
python .github/scripts/release_version.py --tag v<semver>
python .github/scripts/release_version.py --pep440 <pep440>
```

For example, repository release `1.0.0-beta.2` maps to tag and GitHub Release
`v1.0.0-beta.2`, release note `docs/releases/v1.0.0-beta.2.md`, and Python/PyPI
version `1.0.0b2`.

## PyPI Trusted Publishers

Before the first publication with the split workflow, configure all three PyPI
projects:

| PyPI project | Owner | Repository | Workflow | Environment |
|---|---|---|---|---|
| `gafime` | `onlyxItachi` | `GAFIME` | `publish_release.yml` | `pypi` |
| `gafime-cuda` | `onlyxItachi` | `GAFIME` | `publish_release.yml` | `pypi` |
| `gafime-rocm` | `onlyxItachi` | `GAFIME` | `publish_release.yml` | `pypi` |

The workflow value is the filename, while its repository path is
`.github/workflows/publish_release.yml`. Add and verify these entries before
removing the prior `build_wheels.yml` Trusted Publisher entries. After one
successful publication, remove or disable the old entries so only the
manual-only publisher can obtain PyPI credentials.

The publication preflight verifies through PyPI's public API that all three
project identities exist. PyPI does not expose Trusted Publisher bindings
through that API, so confirming the workflow filename and environment for all
three entries remains a manual, blocking pre-tag check.

Do not create PyPI projects or publishers for Metal, RT/OptiX, or bundled ROCm.

## Build And Freeze

`.github/workflows/build_wheels.yml` builds and validates but cannot publish.
Run it from the exact reviewed candidate:

```bash
gh workflow run build_wheels.yml --ref <candidate-ref>
gh run watch <build-run-id> --exit-status
gh run view <build-run-id> --json headSha,conclusion,jobs,url
```

The workflow must:

1. build every manifest-declared dedicated CPython wheel and all three sdists;
2. validate installed Core, CUDA, ROCm, and Apple Metal surfaces;
3. prove archive composition and dependency direction;
4. write checksums and source/run provenance;
5. upload one immutable `release-bundle`.

`core_wheel_build_tag` is a pre-freeze build input for a specifically reviewed
recovery case. Leave it empty for a normal release.

Inspect the frozen bundle:

```bash
gh run download <build-run-id> --name release-bundle --dir dist
python tests/release_measure/artifact_01_release_composition.py \
  --scope full-release --artifacts dist
python .github/scripts/check_pypi_artifact_collisions.py \
  --artifacts dist --version <pep440>
```

The ROCm build also uploads `rocm-wheel-policy-report.json` as evidence outside
the frozen bundle. Schema v2 contains one deterministic size entry for every
manifest-declared CPython wheel and one shared policy identity. Review its
policy hash, per-wheel size totals, `userspace_bundled=false`, truthful platform
tag, and `libamdhip64.so.7` prerequisite.

## Normal Publication

After the reviewed commit is on `main` and the build run succeeds:

1. Confirm the release note and version surfaces are final.
2. Create `v<semver>` on the exact build-run commit and push the tag.
3. Confirm all three Trusted Publisher entries name `publish_release.yml`.
4. Dispatch the publisher with the exact build run and tag:

```bash
gh workflow run publish_release.yml --ref v<semver> \
  -f build_run_id=<build-run-id> \
  -f release_tag=v<semver> \
  -f allow_matching_existing_pypi_files=false
```

The publisher verifies that:

- the build run used `build_wheels.yml` and concluded successfully;
- the tag resolves to the build run's exact SHA and that SHA is on `main`;
- the downloaded bundle's checksums and provenance are unchanged;
- archive composition and SemVer/PEP 440 identity still pass;
- no PyPI filename already exists.

Publication order is fixed:

```text
Core -> CUDA and ROCm -> public exact-version installs -> GitHub Release
```

CUDA and ROCm cannot run before Core succeeds. Public checks install the
released Core and CUDA wheels across the platform/Python matrix, execute Metal
from the public macOS Core wheel, and build/install the public ROCm sdist
against pinned system ROCm. The GitHub Release is created only after all public
checks pass.

The publisher may copy files into per-project upload directories only to select
them. It verifies each selected file is byte-identical to the frozen source.
It must never build, repair, retag, rename, or otherwise mutate a package.

## Hash-Matched Recovery

PyPI files are immutable. Reuse the same build run, tag, and frozen bundle
after a partial publication. First inspect which filenames exist and compare
their SHA-256 values.

Only when every collision is byte-identical, dispatch:

```bash
gh workflow run publish_release.yml --ref v<semver> \
  -f build_run_id=<original-build-run-id> \
  -f release_tag=v<semver> \
  -f allow_matching_existing_pypi_files=true
```

The collision preflight rejects any hash mismatch. Existing identical files are
skipped; missing files are uploaded. This same path recovers a failed
GitHub-Release-only step after all PyPI files exist. Never rebuild a supposedly
identical artifact.

## Abandoned Partial Publication

Use this path only when one or more payload releases reached PyPI, the matching
exact-version Core release did not, and maintainers decide not to finish that
version:

1. Preserve the failed workflow, frozen hashes, and release note.
2. Yank each affected payload release with a reason naming the missing Core.
3. Do not delete files, reuse the version, or upload replacements.
4. Verify Core is absent and every stranded payload file is yanked.
5. Record the action and continue with a new version.

For the aborted `1.0.0b1` checkpoint:

```text
Aborted partial publication: matching gafime==1.0.0b1 Core was not published.
```

Verify live PyPI metadata:

```bash
python .github/scripts/check_pypi_release_status.py \
  --expect-missing gafime==1.0.0b1 \
  --expect-yanked gafime-cuda==1.0.0b1 \
  --expect-yanked gafime-rocm==1.0.0b1 \
  --reason-contains "matching gafime==1.0.0b1 Core was not published"
```

This follows
[PyPI's release-yanking guidance](https://docs.pypi.org/project-management/yanking/)
and [PEP 592](https://peps.python.org/pep-0592/). Yanking preserves
auditability while keeping normal resolvers away from the abandoned version.
