# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.

## Current Target

- Repository/Cargo candidate target: `1.0.0-rc.2`
- Python/PyPI candidate target: `1.0.0rc2`
- Canonical tag target: `v1.0.0-rc.2`
- Phase: RC2 stabilization, exact-candidate qualification, and rollout

The source tree carries the RC2 identity. Creating or updating
`release/v1.0.0-rc.2` does not by itself create a tag or publication.
Live publication state is authoritative on
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI](https://pypi.org/project/gafime/); this file does not duplicate a
moment-in-time commit, workflow, or package-presence result.

## Completed Gates

- The v1 architecture, precision, ABI, package, and public API documentation
  contracts are established and machine checked.
- The authoritative v1 API reference and public-symbol coverage checks are in
  place.
- The pre-RC security policy, private-reporting path, threat model, and
  historical standard-scan baseline are established.
- Beta.2 source and frozen artifacts completed qualification, but the exact
  frozen documentation could not remain truthful when published. Beta.2 is
  therefore retained as an unreleased checkpoint rather than rebuilt solely
  for publication.
- The three bounded input-validation defects are fixed, the public repository
  and documentation routers are established, and the bounded compiler/codegen
  audit found no evidence-backed product change to apply.
- Repository, Cargo, and Python metadata use the canonical RC2 identities.
- RC1 was built from frozen, verified artifacts and is publicly available as a
  prerelease.

## RC2 Branch Preparation

- Cut the planned `release/v1.0.0-rc.2` branch only from a green `main`, under
  the [candidate release-branch policy](release-branches.md).
- Keep stabilization bounded. Use focused pull requests, merge commits,
  current-head AI review, strict required checks, and resolved review threads.
- Land durable fixes on `main` first where practical. An urgent release-first
  fix must be present on `main` no later than final admission; the admission
  merge normally supplies that forward-port. Never merge divergent `main`
  wholesale into the candidate branch.
- Qualify and build/freeze the exact protected release-branch tip; `main` may
  continue independently while that bounded candidate stabilizes.
- After the frozen candidate is accepted, install exact-ref update/deletion
  protection before admission, tagging, or publication. Retain that lock after
  publication; any source fix requires an explicit unlock and new exact build.
- For final admission, cut a temporary branch from current green `main`, merge
  the unchanged settled release tip into it, and submit that integration branch
  to `main` for strict checks and current-head AI review. Never merge divergent
  `main` into the release branch.
- After admission, verify the frozen release tip is an ancestor of `main`, then
  tag that same tip. Before publication, verify that the authorized
  creation-only rule and the separate no-bypass update/deletion rule both cover
  the exact tag, then dispatch the publisher from that tag ref.
- Require the publisher to match the push/workflow-dispatch build branch,
  remote release tip, tag, dispatch/workflow SHA, downstream checkouts, and
  build SHA exactly, with job-local ref rechecks before irreversible uploads.

## RC2 Hardening

The [RC2 release note](v1.0.0-rc.2.md) records the bounded validation,
guidance, and artifact-identity changes. Local tests and patch review do not
replace the reviewed candidate's required hosted checks and artifact gates.

A deep security review has completed for the preceding source checkpoint.
Its finding dispositions and focused remediation evidence belong to the
hardening review; that earlier scan does not establish security qualification
for a later settled candidate. Resident NumPy input now uses privately owned
snapshots to bind content identity to consumed bytes. Acquisition is not atomic
against external writers; coherent datasets still require synchronization
during acquisition. Broader retained/borrowed and low-copy ingestion belongs to
future issue #100, not RC2.

## RC2 Release Gates

The accepted candidate requires current-head PR review and configured checks,
an exact-candidate standard security record, and a complete frozen release
bundle. Required Core/CUDA/ROCm/Metal execution evidence must identify the actual
artifact hashes. The release tip is locked and admitted unchanged to `main`
before canonical tagging; collision and publisher prerequisites are checked
before the official frozen publisher runs. Live workflow and public-channel
results establish completion rather than this source document forecasting it.

No RC2 tag, GitHub Release, or PyPI publication follows merely from branch
preparation. Stable qualification and the permanent performance
architecture tracked by issue #71 remain later work.
