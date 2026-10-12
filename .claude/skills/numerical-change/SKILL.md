---
name: numerical-change
description: Evaluate or implement a GAFIME numerical change across precision routes, metrics, or backends while preserving declared semantics and evidence-backed tolerances.
metadata:
  audience: contributor
---

# Numerical Changes

Resolve the current contract from `docs/precision-contract.md`, the Numerical
Policy in `AGENT.md` or `CLAUDE.md`, and the affected ABI and backend documents.
Do not infer a numerical promise from an implementation detail or an old
benchmark.

Identify every affected lane before editing:

- ingest and resident storage;
- pointwise transformation or interaction materialization;
- reductions and statistics;
- ranking and public result representation;
- cache, compiled, graph, and artifact identity; and
- each supported backend and metric.

Prove semantic correctness against the approved independent oracle. Exercise
finite, non-finite, constant, high-dynamic-range, overflow/underflow, boundary,
and deterministic-order cases relevant to the change. Cross-backend agreement
is not sufficient when every backend could share the same error.

Exact parity is required where the contract requires it. When unavoidable
hardware or compiler behavior prevents exact floating-point parity, retain an
explicitly documented and maintainer-approved tolerance supported by measured
evidence. Performance is not a justification for an undocumented numerical
difference, and fast-math or reassociation behavior must not be introduced
through a compiler flag, intrinsic, or convenience refactor without the
required contract decision.

Apply the contract's performance evidence and boundary-cost rule when numerical
acquisition or execution changes. Preserve independent parity while measuring
the affected components and installed public workflow, including default and
realistic configs, cold/hit behavior, latency/CPU, fresh-process peak RSS and
justified copies. Do not move checked narrowing into Python, lose source NaN/Inf
distinctions, or stage fp64 through fp32 to improve a benchmark. Exact artifact
and input/configuration identities, and combined/not-observable phase labels,
keep cost evidence separate from numerical guarantees.

Report which profiles, metrics, backends, data shapes, and public fields were
validated, along with the oracle and tolerance rationale. If a proposed change
needs a new precision meaning, public selector, ABI route, or fallback policy,
stop treating it as an internal numerical patch and surface the required design
decision.
