---
name: performance-change
description: Evaluate or implement a GAFIME performance change with claim-matched correctness, production, kernel, code-generation, and physical-hardware evidence.
metadata:
  audience: contributor
---

# Performance Changes

Read the active checkout's regression and numerical policy in `AGENT.md` or
`CLAUDE.md`, the relevant execution contract, and
`tests/release_measure/README.md`. Treat historical measurements as context,
not current-head proof.

Match evidence to the claim:

- semantic and numerical tests establish correctness, not speed;
- public end-to-end timing measures the user-visible lifecycle it executes;
- native or kernel timing isolates that layer but does not establish product
  throughput;
- IR, assembly, device ISA, counters, registers, and spills explain generated
  code but do not replace runtime evidence; and
- a GPU performance claim requires execution on the named physical device.

For Core product throughput, exercise the production executor with its normal
candidate-level Rayon parallelism and per-candidate SIMD/native arithmetic.
Single-worker or leaf-kernel measurements are supplemental diagnostics. Do not
substitute them for the production claim.

Compare an exact base and exact candidate under the same controlled environment,
workload, affinity, worker policy, toolchain, and measurement method. Preserve
raw samples, ordering, provenance, and relevant artifact hashes. Establish
correctness before timing, identify contamination or censoring honestly, and do
not repeatedly rerun a campaign merely to obtain a preferred result.

Keep conclusions bounded to the measured shapes, host, device, precision,
metric, and lifecycle. Capability probes, successful compilation, static target
coverage, or a smaller binary are useful facts but are not universal throughput
claims. If an apparent optimization changes numerical, API, ABI, fallback, or
ownership semantics, treat it as the corresponding contract change rather than
explaining the difference away as performance work.

Update durable benchmark infrastructure only when that is the task. Do not turn
a focused optimization into an open-ended benchmark-system redesign.
