# Bounded Polars version / RSS comparison

This is investigative evidence, not a release certificate or general dependency
performance claim. [Run 38214526331](https://github.com/onlyxItachi/GAFIME/actions/runs/38214526331)
compared Polars 1.3.0, 1.28.0, 1.39.0 and 1.44.2 on one Ubuntu 24.04 runner.
No source/runtime rebuild, production allocator override or dependency pin was
used. GAFIME runtime source was PR #111 head
`509d68a99a70182d117dad84eb905dae04764ec1`, still versioned `1.0.0rc2` and not
the public RC2 wheel. Its reused wheel SHA-256 was
`a15feea6a510c7ceb95d88e3aa5dead325bf431a38de6ad917e5c670260dcbb1`; native
SHA-256 was `4ba6e3a767bb7c47ec63771cc0abe6cc6678caa162aba0f36c926f89074327b9`.

## Matched conditions

All versions used the same wheel/native bytes, NumPy 2.4.6 binary, original
Float64 fixtures and fresh-process method. Python was 3.11.17 / GCC 13.3.0,
kernel 6.17.0-1022-azure, glibc 2.39, four eligible CPUs (0–3). System THP was
`[always] madvise never`; inherited allocator/thread overrides were unset.
Raw dependency/native fingerprints and host/runtime receipts are retained.
Polars wheel/runtime archives were verified against public PyPI hashes.

96 uninstrumented public workers and all 48 direct/file pairs passed. Numeric
snapshots were also bit-identical across Polars versions on these workloads.
72 independent uninstrumented producer controls covered all six CSV/Parquet/IPC
× narrow/wide cases; 16 separate phase workers were diagnostic only.
The public execution cases were 100k×20 and 100k×100, fp32/fp64, Core,
Pearson/unary/no significance, cache miss. This is not literal default/full-config
qualification or a comparison of every precision/backend/configuration.

## Results

Maximum observed process HWM across three fresh samples, MiB (1024² bytes):

| Polars | Narrow Parquet public fp32 | Narrow Parquet public fp64 | Narrow Parquet producer | Wide Parquet producer | Wide CSV producer |
|---|---:|---:|---:|---:|---:|
| 1.3.0 | 136.980 | 156.199 | 133.297 | 271.105 | 458.160 |
| 1.28.0 | 158.094 | 174.480 | 157.957 | 290.902 | 369.527 |
| 1.39.0 | 162.539 | 180.031 | 157.016 | 294.391 | 372.973 |
| 1.44.2 | 167.555 | 182.777 | 148.090 | 272.840 | 376.996 |

Narrow public Parquet increased by approximately 30.57 MiB fp32 / 26.58 MiB
fp64 between the endpoints. This is not a monotonic/general RSS regression:
1.44.2 producer Parquet was below 1.28/1.39, wide public Parquet was also below
those versions, and wide CSV producer memory improved by about 81 MiB versus
1.3.0. Direct execution/import peaks differed too; the whole peak difference
must not be assigned to one dataset allocation.

The phase gap was already visible after parsing, before native acquisition.
Across all 16 instrumented trials, precompile RSS → compile HWM added about
7.38–7.55 MiB fp32 or 15.14–15.23 MiB fp64. That is consistent with the required
second precision-typed feature layout, not a version-dependent native clone.
Arrow owners and loader frames are released before compilation; object death
does not promise immediate page return. Cumulative HWM is not additive or an
exact allocation counter, and proc accounting can be approximate.

## Interpretation and disposition

The version screen demonstrates a shape/format-dependent external reader and
allocation/page-policy contribution, but isolates no safely removable GAFIME
allocation or specific upstream bug. It does not justify a dependency downgrade,
private prefetch tweak, allocator override, forced trim, THP change or weaker
ownership. No such product change was applied.

Relevant upstream breakpoints include allocator/background-thread changes,
reader replacement and row-group prefetch policy. These are plausible mechanisms,
not proof of the cause of each byte: [1.28 release](https://github.com/pola-rs/polars/releases/tag/py-1.28.0),
[1.39 prefetch change](https://github.com/pola-rs/polars/pull/26632),
[1.41.2 allocator change](https://github.com/pola-rs/polars/pull/27797).
Polars 1.3.0 already has the Arrow PyCapsule protocol; GAFIME passes the required
`None` argument, so no fake compatibility route was used.

Proceed with [the capped parser-aware RSS model](../ingestion-rss-policy.md).
Its producer controls must pass the original independent RSS/latency ceilings,
and public latency/numerical/ownership/process gates remain unchanged. Retained
original failed verdicts are not retroactively passed. Final qualification uses
the strengthened complete provenance protocol, not this earlier diagnostic
receipt alone.

## Evidence receipt

Artifact `11731446342`, `rc3-polars-version-comparison`, 3,350,820 bytes, SHA-256
`69c4248f97dbf98740bb200b5e2fef3a394765c6f23884db52ccbdaaa1d18900` was independently
downloaded and the complete ZIP digest verified. It retains public/producer
reports, raw samples/logs, phase markers and proc observations, pinned dependency
lists, PyPI metadata and the original Core wheel identity. Original large
fixture payloads remain in the referenced prior artifact rather than being
duplicated. Fixed measurements do not establish identical physical behavior on
a different runner, or certified cross-host performance.

## Independent capped-policy validation

[Run 38216392423](https://github.com/onlyxItachi/GAFIME/actions/runs/38216392423)
then reused the same candidate wheel and the separate public RC2 wheel
`a816fdde6d9836228474fb8d152679c8fba5e5bb6c198f3b522c0e9b69810790`, with
identical pinned Python/runtime/dependency conditions on one runner. It collected
216 uninstrumented public workers (108 paired numerical comparisons) and 36
independent producer workers: all six shape/format cases, three samples, for
each artifact separately. No wheel was rebuilt.

- Candidate: all 18 public latency/RSS cells and all producer RSS/latency guards
  passed the parser-aware model.
- Public RC2: all 18 cells still failed **both** public latency and RSS; its own
  producer controls were used, not a candidate-derived envelope.
- The full inherited/effective allocator, THP and thread receipt was verified.
  All 10 inherited allocator/thread overrides were unset, all four system THP
  fields were captured, and effective Polars policy was
  `dirty_decay_ms:500,muzzy_decay_ms:1000` with observed pool size 4.

For narrow candidate Parquet, peaks were 177,881,088 / 171,696,128 / 190,054,400
bytes for fp32/mixed/fp64. The separately measured parser peak was 157,048,832
bytes, below every original independent cap; the resulting public ceiling was
202,020,352 bytes. Original ceilings remain recorded (174,724,608 bytes for
fp32/mixed and 187,364,864 for fp64), not retrospectively changed.

Artifact `11731608770`, `rc3-parser-budget-validation`, 7,326,596 bytes, has
SHA-256 `d0f79f9c6ee95d5bb34df8eb34f8d299fc2a8a10a88e2aa5b10e16a8167d4ed8`.
Earlier run 38215483658 used the pre-completeness receipt and remains historical
evidence; it does not stand in for the final full-provenance validation.
These investigative results do not replace required exact-final-head PR CI or
final frozen RC3 artifact/source binding, and do not authorize publication.
