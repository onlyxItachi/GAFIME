# Core production benchmark calibration

The common benchmark harness is measurement infrastructure, not product code.
Its source revision is independently authenticated from each measured product.
The [release-measure guide](../tests/release_measure/README.md) defines the full
production, numerical, statistical and provenance contracts.

## Loop selection is not acceptance

Selecting a count that is predicted to take exactly 200 ms leaves almost no
margin above the subsequent 200 ms acceptance boundary. Ordinary warming and
changing worker placement can make a slightly larger count run faster again,
exhausting bounded calibration without recording any samples.

The selector therefore plans toward **220 ms**:

```text
next_count = max(current_count, ceil(current_count * 220_000_000 / observed_ns))
```

Checked arithmetic, positive denominators, representable counts and the loop
ceiling remain mandatory. Excessive counts fail closed; they are not clamped.
The separate selection target is recorded in
`calibration.loop_selection_target_region_ns`.

None of these acceptance requirements changes:

- All three final preflights must reach **200 ms**.
- At most four refinements and 1,048,576 loops are permitted.
- Changing the count restarts the complete warmup phase.
- Recording uses one fixed count, at least 10 warmups and 30 raw samples.
- Every recorded region must reach **100 ms**. Samples are never filtered,
  retried, rescaled or used to adapt the count.
- Numerical parity, statistical intervals, regression margins and the
  distinction between default-worker production and scaling diagnostics remain
  unchanged.

Under a linear per-loop-cost model, the selection margin tolerates a 9.09%
later reduction in per-loop duration. It is not a convergence guarantee, and
integer quantization means runtime cost need not increase uniformly by 10%.

## Bounded RC3 diagnostic evidence

On 2026-10-17 an isolated reproduction of the failed 40th candidate cell used
common-f64 input, fp32 mutual information, 2,048 rows, 48 features and one
worker. Following the preceding parallel workload, the original selector used
481, 734, 739, 740 and 743 loops. Its final four minimum preflights were
198.850181, 199.970333, 199.208499 and 199.546447 ms. It exhausted four
refinements before recording samples.

Boundary snapshots directly observed worker migration and changing execution
rate. They do not isolate migration, clock/thermal behavior, SMT/cache effects
or other host activity as the sole cause. The original Actions failure's
unrecorded round history cannot be reconstructed from this reproduction.

The matched finite diagnostic plans retained every failure:

| Harness | Instrumented trials | Uninstrumented controls |
| --- | --- | --- |
| Original selector | 10 succeeded, 1 calibration failure | 2 succeeded |
| Headroom selector | 11 succeeded | 14 succeeded |

All 37 successful child contracts validated. There were 21 exact same-product
full-result-snapshot comparisons and seven baseline/candidate pairs under the
existing parity oracle. The fixed controls covered all precision profiles and
metrics, with one/default workers in a focused subset, not the full matrix.
The lowest accepted fixed preflight was 207.391119 ms and lowest recorded
region was 204.433441 ms. Negative controls still rejected under-target
preflights, a fifth refinement, excessive loop counts and an under-floor raw
region; exact-200 ms preflights remained accepted.

These were newly built diagnostic rlibs from exact products
`d52199f44aa80ab8ef50c18db95dd1630961cdaf` and
`b510b4dd9af4c47602846b9a58702ed005fe67e4`, using Rust 1.97.1. They were not
byte-identical replays of the sealed campaign executables or installed-wheel
qualification. The same rebuilt libraries were used across each product's
original/fixed comparisons. Instrumentation was outside timed regions but
could perturb execution; separate uninstrumented controls were material.

The validated uninstrumented Rust harness source SHA256 is
`3c04a1d7ab77241f02d93a152de5d23e6ff1b1b9e601685e03241cbff7dfaca6`.
The privately retained diagnostic archive SHA256 is
`5e8d165a6edbbbf48708458ac44d6ed83674b090ed7782edad95da937af81fd2`.
Its manifest, raw traces, host telemetry and independent review remain evidence,
not a formal comparative performance pass or a universal reliability claim.

## Exact-revision recovery contract

A new formal campaign must pin one reviewed, clean common-harness commit
separately from the frozen baseline and candidate product commits. Both
products must use that identical harness, runner and independent validator.
Record their full commits, trees, source blobs/hashes, compiled arguments,
rlibs, executables and wheel identities; do not infer product identity from the
harness checkout's package version.

Preserve allocator, THP, thread, affinity, power and host provenance. Use fresh
output paths and complete matched A/B plus reversed B/A blocks. Reopen and
independently validate the new raw evidence. Do not splice partial candidate
records or pair an old-harness baseline with a revised harness.

Changing only this separately pinned measurement harness does not modify the
frozen product source or authorize rebuilding its wheels. Normal PR review and
required merge-commit CI apply to the harness change. Campaign execution,
release admission, tagging and publication remain separate authorized actions.
