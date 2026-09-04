---
name: backend-change
description: Evaluate or implement a GAFIME Core, CUDA, ROCm/HIP, or Metal change while preserving ownership, ABI, lifecycle, capability, and distribution contracts.
metadata:
  audience: contributor
---

# Backend Changes

Use the active checkout's `AGENT.md` or `CLAUDE.md` as the ownership map. Read
the affected portions of `docs/contract.md`, `docs/abi-evolution.md`,
`docs/backend-selection.md`, `docs/capabilities.md`, and the manifest-derived
release artifact policy rather than copying their mutable details here.

Preserve the boundary:

- Rust owns validation, planning, scheduling, backend selection, and safe public
  lifecycle policy.
- A native backend owns its device kernels, launcher, runtime interaction, and
  backend-local resources behind the approved C ABI.
- Python remains the public declaration/reporting surface rather than a scalable
  data-plane fallback.

Validate ABI-visible pointers, lengths, shapes, strides, counts, enums, flags,
structure versions, and arithmetic before native access. Keep unsafe ownership
and asynchronous lifetimes explicit. Backend-local execution and teardown must
preserve caller device state and avoid sharing mutable device resources across
incompatible devices or lifetimes.

Capability reporting must describe what the loaded payload and selected device
actually support. Explicit unsupported requests fail closed; do not turn a
payload, runtime, precision, or operation failure into silent execution on
another backend. Preserve deterministic candidate/result identity across
parallel or asynchronous execution.

For validation, separate source/build coverage, exported ABI inspection,
installed-package behavior, and physical device execution. Compilation and
static architecture targets do not prove runtime execution. Test the public API,
the native boundary, affected lifecycle paths, numerical parity, error
propagation, and artifact/package composition proportionately.

Check the live release manifest before making distribution claims. Experimental
or local-only paths must remain isolated unless the task explicitly authorizes
a reviewed promotion. A change that needs another ownership model, public ABI,
package identity, or fallback policy is an architecture decision, not a bounded
backend patch.
