# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.

## Current Target

- Current public release: `v1.0.0-rc.1` (`1.0.0rc1` on PyPI)
- Next repository/Cargo candidate target: `1.0.0-rc.2`
- Next Python/PyPI candidate target: `1.0.0rc2`
- Next canonical tag target: `v1.0.0-rc.2`
- Phase: RC2 release-branch preparation; no RC2 tag or release

The source tree continues to carry the RC1 identity until a focused RC2
preparation change is reviewed and merged. Creating
`release/v1.0.0-rc.2` does not by itself change that identity, create a tag, or
create a release. Live publication state is authoritative on
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
- Repository, Cargo, and Python metadata use the canonical RC1 identities.
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
- For final admission, cut a temporary branch from current green `main`, merge
  the unchanged settled release tip into it, and submit that integration branch
  to `main` for strict checks and current-head AI review. Never merge divergent
  `main` into the release branch.
- After admission, verify the frozen release tip is an ancestor of `main`, then
  tag that same tip and publish only its byte-identical frozen bundle.
- Require the publisher to match the push/workflow-dispatch build branch,
  remote release tip, tag, and build SHA exactly; after publication retain the
  exact release branch under read-only protection.

No RC2 tag, GitHub Release, or PyPI publication exists merely because branch
preparation has begun. Stable qualification, the Deep Security Scan, and the
permanent performance architecture tracked by issue #71 remain later work.
