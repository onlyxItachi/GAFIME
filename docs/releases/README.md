# Releases

This index records public releases and intentionally retained checkpoints.
[CHANGELOG.md](../../CHANGELOG.md) describes changes chronologically;
[STATUS.md](STATUS.md) is mutable current-train state; each versioned note is
the historical record for that release or checkpoint.

## Current Release Train

Current source prepares stable `v1.0.0` from post-RC3, post-pruning mainline.
Stable identity is not completed qualification or publication. The obsolete
unpublished stable attempt is preserved under an immutable archival reference;
its canonical name is available for a new fork only after approval and green
stable-prepared `main`. See [stable source integration](stable-integration.md).

Published [RC3](v1.0.0-rc.3.md) remains an immutable provenance record. Stable
requires a new exact-source candidate, qualification, verified frozen artifacts
and final admission of its unchanged tip to `main` under the existing runbook.
Source preparation does not itself create a tag or release. Follow the mutable
[release status](STATUS.md) for current gates and the live
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI project](https://pypi.org/project/gafime/) for publication state.

The [stable preparation note](v1.0.0.md) describes the intended source scope;
live release presence is established by the public channels above.

## Release Operations

- [Release operations runbook](release-operations.md)
- [Candidate release-branch policy](release-branches.md)
- [Manifest-derived artifact matrix](release-artifact-matrix.md)

## Release History

### v1

- [`v1.0.0`](v1.0.0.md) — stable source preparation; this entry does not establish
  completed stable qualification or publication.
- [`v1.0.0-rc.3`](v1.0.0-rc.3.md) — published release candidate with native file
  acquisition hardening and qualified immutable artifacts.
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
