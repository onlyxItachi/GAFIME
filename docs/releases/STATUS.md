# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.

## Current Target

- Repository/Cargo candidate target: `1.0.0`
- Python/PyPI candidate target: `1.0.0`
- Canonical tag target: `v1.0.0`
- Phase: stable stabilization, exact-candidate qualification, and rollout

This source line carries the stable identity. Durable fixes must first land
on `main` through their reviewed PRs; the identity preparation follows through
its own PR. Creating or updating `release/v1.0.0` does not by itself create a
tag or publication, complete a security review, or qualify frozen artifacts.
Live gate and publication results are authoritative in
[GitHub Actions](https://github.com/onlyxItachi/GAFIME/actions),
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI](https://pypi.org/project/gafime/); this file does not duplicate a
moment-in-time commit, workflow, or package-presence result.

## Established Contracts And Prior Evidence

These records do not complete the final stable-candidate gates below.

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
- Repository, Cargo, and Python metadata now agree on the canonical stable
  target; this is source preparation, not publication evidence.
- RC1 was built from frozen, verified artifacts and is publicly available as a
  prerelease.

## Stable Branch Preparation

- Cut the planned `release/v1.0.0` branch only from a green `main`, under
  the [candidate release-branch policy](release-branches.md).
- Prepare stable identity through its own reviewed PR into that protected
  candidate branch; the later admission PR brings it to `main` unchanged.
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

## Stable Stabilization Scope

The [stable release note](v1.0.0.md) records checked file-ingest and seed parity,
adaptive time-series significance and compiled metadata retirement, uniform
Rust configuration checks, and ordinary CUDA/HIP native-call coordination.
The [RC2 release note](v1.0.0-rc.2.md) remains its unchanged historical record.
Local tests, physical candidate runs, and patch review do not replace the final
reviewed candidate's required hosted checks, security record, and artifact gates.

Earlier security records, finding dispositions, and focused remediation
evidence remain bound to their original source checkpoints; they do not
establish final stable-candidate qualification. Resident NumPy input uses
privately owned snapshots to bind content identity to consumed bytes. Acquisition is not atomic
against external writers; coherent datasets still require synchronization
during acquisition. Broader retained/borrowed and low-copy ingestion belongs to
future issue #100, not stable v1.0.0. GIL release, TLS/cache ownership changes,
Rayon topology, and shutdown redesign are outside this stabilization scope.

## Stable Release Gate Sequence

- Land durable fixes on `main` and stable identity on the candidate branch
  through focused PRs with current-head AI review, configured checks, and
  resolved review conversations.
- Complete required hosted correctness checks and final exact-candidate deep
  Codex Security qualification with retained finding dispositions; earlier
  scans do not qualify this later source tip.
- Build, verify, and freeze the complete manifest-derived release bundle from
  that settled tip; local candidate artifacts are not the final frozen bundle.
- Then execute the required Core/CUDA/ROCm/Metal gates against exact members
  of that verified frozen bundle, retaining their hashes and physical execution
  evidence separately from hosted compilation and local candidate runs.
- Lock the release tip and admit it unchanged to `main` before canonical
  tagging; verify tag protections, collision checks, and publisher prerequisites.
- Run the official frozen publisher in Core-first order and verify public
  exact-version installations before the GitHub Release is created.

Live workflow and public-channel results establish completion rather than this
source document forecasting it. No stable tag, GitHub Release, PyPI publication,
or completed final security qualification is claimed by this preparation.
The permanent performance architecture tracked by issue #71 remains separate
future work; no universal throughput or comparative GPU speedup is claimed.
