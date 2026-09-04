# Releases

This index records public releases and intentionally retained checkpoints.
[CHANGELOG.md](../../CHANGELOG.md) describes changes chronologically;
[STATUS.md](STATUS.md) is mutable current-train state; each versioned note is
the historical record for that release or checkpoint.

## Current Release Train

`v1.0.0-rc.1` is the current public prerelease. The next candidate target is
`v1.0.0-rc.2`; its release-branch preparation does not itself change the source
version or create a tag or release. Its protected settled branch tip will be
the candidate source while `main` may continue; publication still requires
final admission of that exact tip into `main`. Follow the mutable
[release status](STATUS.md) for current gates and the live
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI project](https://pypi.org/project/gafime/) for publication state.

## Release Operations

- [Release operations runbook](release-operations.md)
- [Candidate release-branch policy](release-branches.md)
- [Manifest-derived artifact matrix](release-artifact-matrix.md)

## Release History

### v1

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
