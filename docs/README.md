# GAFIME Documentation

Choose the path that matches what you are trying to do. The root
[README](../README.md) is the project landing page; this index routes to the
documents that own each subject.

## Getting Started

- [Guided v1 tutorial](notebooks/gafime_tutorial.ipynb) — an approachable first
  analysis and the main workflows.
- [Practical usage guide](../USAGE.md) — concise configuration and operational
  guidance.
- [Authoritative v1 API reference](notebooks/gafime_v1_api_reference.ipynb) —
  the complete public API and cookbook.
- [Public API coverage](public-api-coverage.md) — machine-checked symbol and
  example coverage.

## Execution

- [Backend selection](backend-selection.md)
- [Capability reporting](capabilities.md)
- [Precision profiles](precision-contract.md)
- [Eager, resident, and compiled lifecycles](eager-resident-compiled-execution.md)
- [ROCm distribution policy](rocm-wheel-policy.md)

## Maintainer Architecture

- [Normative v1 contract](contract.md)
- [ABI evolution and compatibility](abi-evolution.md)
- [Correctness-boundary hardening](correctness-boundary-hardening.md)
- [CPU fused continuous accumulation](cpu-fused-continuous-accumulation.md)
- [CUDA template-kernel hardening](cuda-template-kernel-hardening.md)
- [GPU target-statistics cache](gpu-continuous-target-stats-cache.md)
- [GPU Spearman target-rank cache](gpu-spearman-target-rank-cache.md)

### Design Experiments (Not Shipping APIs)

- [Issue #73 native evidence feasibility](issue-73-native-evidence-feasibility.md)
  — a bounded post-v1 probe of candidate/evidence reuse and the #72 dependency.

## Build and Contribution

- [Source builds and toolchains](../BUILD.md)
- [Contribution and review governance](../CONTRIBUTING.md)
- [Agent repository contract](../AGENT.md)
- [Agent skill audiences and bootstrap](agent-skills.md)

## Security

- [Security policy and private reporting](../SECURITY.md)
- [Repository threat model](security/threat-model.md)
- [Historical pre-RC baseline](security/pre-rc-baseline.md)

## Releases

- [Current release-train status](releases/STATUS.md)
- [Release index and history](releases/README.md)
- [Candidate release-branch policy](releases/release-branches.md)
- [Release operations runbook](releases/release-operations.md)
- [Release artifact matrix](releases/release-artifact-matrix.md)
- [Chronological changelog](../CHANGELOG.md)

## Evidence and History

- [Evidence index](evidence/README.md)
- [Historical pre-v1 API notebook](notebooks/gafime_full_api_reference_notebook.ipynb)
- [Historical release records](releases/README.md#release-history)
- Historical investigations and platform records remain under `docs/`; their
  original claims are evidence for the source state they describe, not current
  architecture unless a normative document links them as such.
