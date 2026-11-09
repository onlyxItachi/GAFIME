# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.
On a work or candidate branch it describes that branch's preparation; identity
does not assert that `main` has adopted an unmerged change.

## Current Target

- Prepared repository/Cargo and Python/PyPI identity: `1.0.0`
- Canonical future tag target: `v1.0.0`
- Phase: post-RC3, post-pruning stable source preparation; final qualification pending

Stable identity and Production/Stable metadata describe the intended release
line, not a public stable distribution or an accepted frozen artifact set.
The earlier unpublished stable attempt has been archived without rewriting its
history. The replacement `release/v1.0.0` must be cut only after the focused
source-preparation PR is approved by the maintainer, merged, and green on
`main`. No replacement candidate is established by this document. See the
[stable integration record](stable-integration.md).

Live publication state is authoritative on
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI](https://pypi.org/project/gafime/). This file does not embed an exact current
commit, workflow result or artifact count.

## Established Architecture And Earlier Evidence

- RC1, RC2 and RC3 are public prereleases. Their protected release branches,
  tags, package bytes and frozen provenance remain immutable.
- RC3 completed exact-source standard security qualification, hosted checks,
  frozen-bundle verification, installed Core/public-ingest E2E checks, formal
  Core benchmarking, applicable physical backend gates, final admission and
  public-install verification. Evidence retains its original source, artifact,
  hardware, harness and disclosed limitations; it is not stable qualification.
- Post-RC3 mainline retains the reviewed Core calibration-headroom fix and the
  focused macOS Rustup CI repair. Neither is replaced by the old stable attempt.
- The four reviewed pruning changes remove obsolete configuration mappings,
  Python compiled projections/export handles and scheduler utilities; they
  retain modern compiled lifecycle, real Arrow export, public integrations,
  deprecated-but-available DataQualityAnalyzer and unchanged OTSEncoder.
  See [the stable note](v1.0.0.md) and [API disposition](../legacy-pruning.md).
- Default numeric `dataload` retains incremental checked native Arrow acquisition
  with full configuration, owned cache/execution identity and early foreign-owner
  release. Parser-aware RSS, latency, numerical parity and complete
  allocator/THP/thread provenance remain separate public-workflow gates.
- Native planning, scheduling, candidate identity/order, CPU/GPU execution,
  numerical profiles, ABI 1.0/1.1, package topology and supported CPython matrix
  are unchanged. Core/CUDA/ROCm support fp32/mixed/fp64; embedded Metal is fp32
  only and unsupported explicit profiles fail closed.
- Earlier security reviews remain evidence for their original checkpoints,
  not a Deep Security Scan of the settled stable candidate.
- Beta.2 remains an unreleased pre-RC checkpoint. Abandoned beta.1 payloads are
  historical yanked files, not a matching-Core release.

## Remaining Stable Gates

1. Complete the focused source-preparation PR's independent exact-head AI
   review, strict current head/base CI and blocking-thread resolution; obtain
   the maintainer's final approval before merge.
2. Verify green stable-prepared `main`, then fork and normally protect the new
   canonical `release/v1.0.0`. Record and independently review the settled tip.
   Never merge divergent `main` wholesale into an established candidate.
3. After separate authorization, run Deep Security Scan on that exact settled
   stable source. Remediate and revalidate release-blocking findings; source
   changes require reconsidering or repeating exact-source evidence.
4. Complete required installed-package, ABI/numerical, public-ingestion/E2E,
   Core production and final DL Method evidence. Explicitly assess any reusable
   evidence rather than silently relabeling RC3 qualification as stable.
5. Build/freeze a new exact-release-tip, manifest-derived artifact set; verify
   provenance, checksums, dependencies, composition and RT exclusion. Qualify
   the actual frozen Core/CUDA/ROCm/Metal artifacts with exact hashes.
6. After acceptance and authorization, lock the exact tip and admit it unchanged
   to `main`. Refresh collision, Trusted Publisher, beta.1 resolver-safety and
   immutable-ref prerequisites before separately authorized canonical tagging
   and official Core-first frozen publication.

Follow [candidate release branches](release-branches.md),
[release operations](release-operations.md) and the
[artifact matrix](release-artifact-matrix.md). No pending gate is marked passed
by a source version, classifier or this checklist.

The accepted candidate requires exact-ref update/deletion protection before
admission, tagging, or publication; retain it after publication, and any source
fix requires an explicit unlock and new exact build. Verify both the authorized
tag-creation rule and the separate no-bypass update/deletion rule. The official
publisher retains job-local ref rechecks before irreversible uploads.

## Deliberately Deferred

Issue #71's permanent benchmark architecture remains open. GIL detachment,
cancellation, broader borrowing/ownership (#100), Polars 2 migration (#87), OTS
modernization (#119) and future feature/RT work remain outside preparation.
Analysis still holds the GIL; owned acquisition is not atomic against external
writers. No new runtime API, numerical route, package family, security guarantee
or universal performance claim is introduced. A dedicated v0.x-to-v1 migration
notebook remains pending; the pre-v1 full API notebook stays historical.
