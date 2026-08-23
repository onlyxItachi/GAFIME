# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.

## Current Target

- Repository/Cargo target: `1.0.0-rc.1`
- Python/PyPI target: `1.0.0rc1`
- Canonical tag target: `v1.0.0-rc.1`
- Phase: pre-RC source qualification and presentation cleanup

The source tree retains beta.2 version metadata until the focused RC identity
change is reviewed and merged. Live publication state is authoritative on
[GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases) and
[PyPI](https://pypi.org/project/gafime/).

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

## Remaining RC1 Gates

- Merge the bounded validation fixes and repository presentation changes.
- Complete bounded correctness and compiler/codegen qualification.
- Merge the exact RC1 version and release record.
- Run and retain a standard Codex Security scan against the exact RC1 source;
  remediate and rescan any release-blocking finding.
- Pass exact-source V1/native validation and Build and Validate Wheels.
- Independently verify the frozen bundle and contracted physical backend
  execution.
- Complete release collision, abandoned-beta resolver-safety, and Trusted
  Publisher prerequisites.
- Bind the canonical tag to the verified source, publish the frozen bytes, and
  verify public exact-version installations.

Stable qualification, the Deep Security Scan, and the permanent performance
architecture tracked by issue #71 remain later work.
