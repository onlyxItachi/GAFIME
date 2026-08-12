# Pre-RC Security Baseline

This record describes the practical security baseline performed before the
first GAFIME v1 release candidate. It does not claim that GAFIME is
vulnerability-free, and it is not the later deep qualification planned for the
RC-to-stable transition.

## Scope and identity

- Baseline revision: `df4ab197e33b761b4c7236ea2e1642d90fac1357`
  (post-PR #70 `main`).
- Review mode: one standard, repository-wide Codex Security scan.
- Scan ID: `54877e12-9db4-4ea4-98dd-d0f72910acd6`.
- Snapshot digest:
  `codex-security-snapshot/v1:sha256:32b255bfaa1f64431dc627d61fbecccc5e0f6ea1d8a67fd0e0b322bcfd67ce0b`.
- Deep Security Scan: not run; reserved for the later stable-release
  qualification.

The review covered the Python/PyO3/Rust boundaries, unsafe Rust, SIMD and raw
ABI execution, ABI 1.0 and 1.1 validation, future-compatible ABI records,
CUDA/ROCm/Metal ownership and device state, payload discovery and caches,
experimental RT isolation, archive validation, release workflows, frozen
artifacts, checksums, and provenance.

## Finding disposition

The standard scan reported one High and three Medium findings. This security
baseline change closes all four with bounded fixes and regressions:

| Severity | Area | Disposition |
| --- | --- | --- |
| High | Safe Rust raw-ABI execution boundary | Raw descriptor consumers now require explicit `unsafe` contracts; owner-backed Python and Rust paths retain narrow audited calls. |
| Medium | Release archive member validation | Tar and ZIP members now require canonical, unique, regular paths and the exact expected sdist root. |
| Medium | CUDA/ROCm caller device state | Standard backend entry points now restore the caller's prior runtime device on every exit. |
| Medium | Publication workflow inputs | Tag and run inputs now cross into shell through quoted environment bindings with strict ref and numeric validation. |

No Critical finding was reported. The scan did not identify a separate
reportable issue in the Python/Arrow boundary, ABI route validation, payload
discovery, resident cache identity, asynchronous lifetime handling, RT package
isolation, or frozen-bundle chain beyond the remediated findings above.

## Validation boundary

Regression evidence includes compile-time unsafe-API assertions, Rust crate
tests, adversarial tar/ZIP tests, publication-workflow source checks, release
composition validation, architecture source gates, and real CUDA and ROCm ABI
smoke builds. The available host exposes one CUDA and one HIP device, so the
multi-device restore behavior is enforced by scoped-guard source checks and
single-device lifecycle execution rather than a physical two-device test.

GitHub Private Vulnerability Reporting is enabled for the repository. Private
reports must use the process documented in [`SECURITY.md`](../../SECURITY.md).
