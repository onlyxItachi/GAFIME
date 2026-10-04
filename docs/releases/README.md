# Releases

This index records public releases and intentionally retained checkpoints.
[CHANGELOG.md](../../CHANGELOG.md) describes changes chronologically;
[STATUS.md](STATUS.md) is mutable current-train state; each versioned note is
the historical record for that release or checkpoint.

## Current Release Train

The current source targets stable `v1.0.0`. Durable stabilization fixes must
land through reviewed PRs before the stable identity is admitted and a candidate
branch is cut from green `main`. The eventual protected settled branch tip is
the candidate source; publication requires exact-tip qualification, a verified
frozen bundle, and final admission of that unchanged tip into `main`.
Source version preparation does not itself complete those gates or create a
tag or release. Follow the mutable
[release status](STATUS.md) for current gates and the live
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI project](https://pypi.org/project/gafime/) for publication state.

The [stable release note](v1.0.0.md) describes the bounded stabilization scope;
live release presence is established by the public channels above.

## Release Operations

- [Release operations runbook](release-operations.md)
- [Candidate release-branch policy](release-branches.md)
- [Manifest-derived artifact matrix](release-artifact-matrix.md)

## Release History

### v1

- [`v1.0.0`](v1.0.0.md) — stable source identity and bounded stabilization;
  exact-candidate qualification and publication remain gated by current status.
- [`v1.0.0-rc.2`](v1.0.0-rc.2.md) — validation and artifact-identity hardening;
  see current status and public channels for publication state.
- [`v1.0.0-rc.1`](v1.0.0-rc.1.md) — public release candidate.
- [`1.0.0-beta.2`](v1.0.0-beta.2.md) — unreleased pre-RC checkpoint; its frozen
  artifacts were qualification evidence, not a public release.
- [`v1.0.0b1`](v1.0.0b1.md) — aborted packaging checkpoint with stranded
  payload publications and no matching Core release.
- [`v1.0.0b0`](v1.0.0b0.md) — public beta.
- [`v1.0.0a0`](v1.0.0a0.md) — public alpha.

### Legacy

- [`v0.5.0-legacy`](v0.5.0-legacy.md) — GitHub-only architecture checkpoint.
- [`v0.4.7`](v0.4.7.md)
- [`v0.4.6`](v0.4.6.md)
- [`v0.4.5`](v0.4.5.md)
- [`v0.4.1`](v0.4.1.md)
- [`v0.4.0`](v0.4.0.md)

Historical records are not rewritten to match current v1 architecture.
