# Current Release-Train Status

This file tracks the current development/release train on `main`. It is mutable
operational status, not an immutable historical release record.

## Current Target

- Repository/Cargo target: `1.0.0-rc.1`
- Python/PyPI target: `1.0.0rc1`
- Canonical tag target: `v1.0.0-rc.1`
- Phase: RC1 qualification and release soak

The source tree carries the RC1 identity. Live publication state is
authoritative on [GitHub Releases](https://github.com/onlyxItachi/GAFIME/releases)
and [PyPI](https://pypi.org/project/gafime/); this file does not duplicate a
moment-in-time tag, workflow, or package-presence result.

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

## RC1 Admission Gates

- Run and retain a standard Codex Security scan against the exact RC1 source;
  remediate and rescan any release-blocking finding.
- Pass exact-source V1/native validation and Build and Validate Wheels.
- Independently verify the frozen bundle and contracted physical backend
  execution.
- Complete release collision, abandoned-beta resolver-safety, and Trusted
  Publisher prerequisites.
- Bind the canonical tag to the verified source, publish the frozen bytes, and
  verify public exact-version installations.

The live links above determine which admission gates have completed. Stable
qualification, the Deep Security Scan, and the permanent performance
architecture tracked by issue #71 remain later work.
