# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.
On a candidate branch it tracks that branch's preparation; candidate identity
does not assert that independently advancing `main` has adopted it before admission.

## Current Target

- Repository/Cargo candidate identity: `1.0.0-rc.3`
- Python/PyPI candidate identity: `1.0.0rc3`
- Canonical next tag target: `v1.0.0-rc.3`
- Phase: RC3 identity preparation and bounded stabilization; final qualification pending

This candidate source carries the RC3 identity on `release/v1.0.0-rc.3`, cut
directly from reviewed green `main`. The initial fork is not a settled candidate;
identity preparation and exact-wheel qualification remain distinct. No final
release freeze, tag or publication follows from this preparation. The unpublished stable
candidate is an earlier checkpoint, not the source to publish as stable now.
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
- The candidate's repository, Cargo and Python metadata use canonical RC3
  identities; published RC2 metadata and artifacts remain unchanged.
- RC1 was built from frozen, verified artifacts and is publicly available as a
  prerelease.
- RC2 is also publicly available from its immutable qualified artifacts. Its
  resident/kernel evidence does not establish public file-ingestion performance.
- Native full-config ingestion hardening has landed through reviewed mainline
  work with installed-wheel numerical/error and latency/RSS gates. Independent
  parser controls preserve complete environment identity and reject the original
  RC2 materialization regression. This is source-hardening evidence, not final
  RC3 frozen-artifact qualification.

## RC3 Boundary Contract And Evidence

The [ingestion hardening plan](../rc3-ingestion-hardening.md) records the defect,
minimal owned native acquisition boundary, copy ledger and acceptance matrix.
Default numeric `dataload` uses checked native acquisition without whole-dataset
Python row/scalar materialization and preserves full configuration. Independent
numerical parity, installed-wheel tests, matched latency/fresh-process memory
samples and applicable backend routing remain required for final candidate
qualification. Every kernel, boundary and affected public workflow must carry
its own evidence; resident throughput is not an end-to-end claim. The
[parser-aware RSS policy](../ingestion-rss-policy.md) preserves independent
producer caps and all existing numeric allowances and latency limits.

The new RC3 branch was cut from green reviewed `main`; canonical identity is
prepared through its focused PR. No old RC2 or stable frozen bundle
qualifies changed RC3 source. Issue #71 stays open; broader Polars 2/combined
streaming and borrowed-memory architecture remain post-v1 work.

## Candidate Branch Preparation

- `release/v1.0.0-rc.3` was cut directly from green `main` under the
  [candidate release-branch policy](release-branches.md). Active protection
  requires merge-commit PRs, strict checks and resolved threads, and prevents
  force-push and deletion without a bypass.
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

## Historical RC2 Hardening

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

## Remaining RC3 Qualification

- Merge the focused identity preparation and any bounded stabilization through
  exact-head review and strict current head/base checks; settle the candidate tip.
- Record the standard security scan against that exact RC3 source, with no
  unresolved Critical/High or release-blocking Medium finding. Earlier security
  checkpoints do not replace it.
- Retain installed public-workflow, numerical/ownership and Core production
  evidence, including defaults, realistic configuration and cache states.
- After separate final-phase authorization, build the exact release tip and
  independently verify its frozen bundle, provenance, checksums, manifest
  composition, dependency/profile identities and RT exclusion. Normal CI
  validation bundles are not an accepted release freeze.
- Qualify the actual artifacts on Core/CUDA/ROCm/Metal as required by the
  runbook, with exact hashes; hosted CUDA/ROCm builds are not physical execution.
- Only after acceptance, lock the tip and admit it unchanged to `main`, then
  recheck collision, publisher, abandoned-beta resolver safety and ref identity
  prerequisites before authorized canonical tagging and official publication.

Live evidence establishes completion; this status does not forecast it.
Published RC2 stays immutable. Stable qualification, GIL detachment and the
permanent benchmark architecture tracked by issue #71 remain later work.
