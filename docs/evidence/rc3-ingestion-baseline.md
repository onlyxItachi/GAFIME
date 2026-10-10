# RC3 public-ingestion baseline investigation

Stable qualification was paused for RC3 after an installed RC2 investigation
confirmed that a fast resident executor did not imply fast public file ingest.
These are preliminary component/end-to-end observations and a matched local
development comparison, not release-certified benchmarks, universal speedups,
or approval of a frozen RC3 artifact.

## Authenticated baseline

- Public Core: `gafime==1.0.0rc2`, CPython 3.14.3, Linux x86_64.
- Wheel SHA-256: `74e6b4f13b73cb9fd769da446b6cbf8b646854a58cd9028dd3253cc0fc6785a8`.
- Native SHA-256: `871311e004bd78a1db0acee9670ef73965673df7c49af60ff7124d03ce80e9c4`.
- The wheel hash was re-queried from public PyPI, and all 28 installed Python
  and native package members were byte-compared with the wheel.
- NumPy 2.5.3; Polars 1.44.2. No baseline wheel was rebuilt or altered.

Synthetic source values were Float32 normal variates from NumPy RNG seed 17.
The target was `0.7*X[:,0] - 0.2*X[:,1] + Normal(0,0.1)`, narrowed to Float32.
Large analysis cells used mixed precision, Pearson, unary candidates, zero
permutations, one repeat, and the normal bounded resident cache. `backend=auto`
resolved to Core under explicitly unavailable vendor payload paths set only in
each child process. A separate small 256 x 8 cell retained literal
`EngineConfig()` defaults, including all metrics and significance.

## Initial uninstrumented observations

Each sample used a fresh isolated process. Fixture generation was separate.
Direct/coercion inputs were preloaded; public `dataload` included file reading.

| Case | Wall time | Process peak RSS |
|---|---:|---:|
| 100k x 20 direct contiguous analysis, 3 samples | median 28.8 ms (18.0–34.5 ms) | median 96.5 MiB |
| 100k x 20 configured Parquet `dataload`, 3 samples | median 658 ms (626–672 ms) | median 184.0 MiB |
| 100k x 100 contiguous coercion, 3 samples | median 11.5 ms (11.2–18.8 ms) | median 141.1 MiB |
| 100k x 100 row coercion, 3 samples | median 2.77 s (2.70–2.78 s) | median 551.0 MiB |
| 100k x 100 configured Parquet `dataload`, 1 sample | 3.13 s | 552.5 MiB |
| 100k x 20 configured CSV / IPC, 1 sample each | 698 ms / 693 ms | 219.2 MiB / 173.5 MiB |
| 100k x 20 restricted explicit-Core Arrow shortcut, 1 sample | 23.3 ms | 108.7 MiB |

The 100k x 100 feature buffer is 38.15 MiB. Row-coercion RSS started around
127 MiB; its peak reached about 551 MiB. Process totals include runtime,
Polars, input and native memory: the excess is not assigned byte-for-byte to
one allocation. All 20-column paths had identical complete interaction-result
digests (identities, order, metrics and candidate diagnostics), excluding
report-level transport text. This is not an independent numerical oracle.

## Attribution and limitations

The literal-default small cell materialized its complete row generator into a
Python list before reconstructing numeric storage. A separate instrumented
20-column cell spent about 698 ms in row coercion, versus roughly 10 ms reading,
4.9 ms in native compile, and 1.7 ms in artifact analysis. These nested,
instrumented times are inclusive and non-additive. Native calls combine phases;
this investigation did not invent isolated native/kernel timings.

Profiling the wide coercion recorded 10.1 million scalar append/validation calls
and 121.8 million total Python calls. Its profiled 17.4 s duration is measurement
overhead, not the uninstrumented 2.77 s baseline. Source confirms the row-object,
flat-array, transport-bytes, decoded-native-vector chain.

AC was online, all 24 logical CPUs were eligible, and no worker/affinity cap was
applied. Moderate desktop activity was present, files were warm in the page
cache, and only a few samples were collected. Peak RSS used fresh-exec Linux
`/proc/self/status` VmHWM: this launch environment inherited an unrelated
`resource.ru_maxrss` high-water mark, which was excluded rather than attributed
to GAFIME. This initial baseline did not measure an RC3 candidate or accelerators.

Raw investigation files remain in local scratch under
`gafime-rc3-rc2-repro.OUIO8n` (`identity.json`, `campaign-summary.json`, per-sample
JSON/logs and profiler output). This record is intentionally preliminary;
acceptance requires the permanent harness against matched installed wheels.

## Permanent reproduction path

`tests/release_measure/perf_14_public_ingest.py` generates fixtures outside
measured workers, binds installations to supplied wheel bytes, records result
bits/significance/diagnostics, and measures fresh-process wall/CPU/RSS. It also
supports explicit cache states and separately labelled component observation.
For example, using operator-supplied installed environments and frozen wheels:

```bash
python tests/release_measure/perf_14_public_ingest.py \
  --python rc2=/env/rc2/bin/python --wheel rc2=/artifacts/rc2.whl \
  --expected-version rc2=1.0.0rc2 \
  --python candidate=/env/candidate/bin/python \
  --wheel candidate=/artifacts/candidate.whl \
  --expected-version candidate=1.0.0rc3 \
  --shape 100000x20 --shape 100000x100 --format csv,parquet,ipc \
  --precision fp32,mixed,fp64 --workflow light \
  --cache-state miss,hit,disabled,target-change --repeats 3 \
  --output /scratch/rc3-ingest-comparison
```

Run `default`, `configured`, `time-series` and `decision-path` separately on
small bounded shapes. The harness preserves failures and raw samples, reports
its comparison exclusions, and does not fabricate an absolute performance
threshold before matched RC3 evidence exists. Independent correctness gates
remain mandatory.

Same-variant direct-vs-file snapshots must match, with missing/duplicate samples
or unequal numeric bits failing the collector. Cross-release equality is not
required because earlier RC2 correctness defects were fixed subsequently.
Collection does not mean release readiness: the output explicitly records
`performance_budget_status="not_assessed"` until matched evidence establishes
approved cost budgets. Generated-family eager paths do not use the continuous
resident cache and accept only `miss`/`disabled` states here.

## Matched installed development candidate

The subsequent `reference-matrix/` campaign compared unchanged public RC2 with
an installed development wheel containing the proposed native acquisition.
The development wheel still reports `1.0.0rc2`; it is **not** the public RC2
artifact or a frozen RC3 release. Its source was uncommitted, so the collector
correctly records its source SHA as null rather than misattributing it to the
base commit. Formal CI and release evidence must bind to an exact reviewed SHA.

- Candidate wheel SHA-256:
  `b7bd7d0c519c305102e4ca14e68910ad609c9df9cc67540c40e44556a7b6745e`.
- Both environments used CPython 3.14.3, NumPy 2.5.3 and Polars 1.44.2.
- Input source: Float64 Gaussian features, RNG seed 17; target
  `0.7*X[:,0] + Normal(0,0.1)`. CSV's direct oracle used the actual parsed values.
- Shapes: 100k x 20 and 100k x 100; CSV/Parquet/IPC; fp32/mixed/fp64;
  `auto` resolving to Core, Pearson/unary/no significance/one repeat,
  cold resident cache. These ingestion-isolation configurations are **not**
  literal `EngineConfig()` defaults.
- 216 fresh-process trials; three samples per variant/route/cell; randomized
  serial order; 108 direct-vs-file numeric comparisons passed. The numeric
  scope includes result bits, identities, order, rankings, diagnostics,
  significance and signal decisions; transport text/backend diagnostics are
  retained separately. This paired check is not an independent numerical oracle.

Selected mixed-profile Parquet observations from that matrix:

| Shape | RC2 `dataload` median wall / median peak RSS | Candidate `dataload` median wall / median peak RSS |
|---|---:|---:|
| 100k x 20 | 651.3 ms / 203.8 MiB | 46.9 ms / 120.4 MiB |
| 100k x 100 | 2966.0 ms / 631.4 MiB | 164.2 ms / 245.1 MiB |

RSS includes imports and source/native storage, and excludes report export
performed after the measured call. AC was online; the normal Rayon worker
configuration was not capped; moderate desktop activity and warm filesystem
cache were present. No competing compilation/test campaign ran during timing.
The reference matrix is local investigative evidence, not a cross-host SLA.

A separate instrumented pilot observed the wide candidate's read/parse boundary
at about 79 ms, native acquisition at 50 ms, compile at 20 ms and artifact
analysis at 3 ms. These wrapper times are inclusive, non-additive observations,
not isolated kernel/planner/allocator measurements. The old row coercion took
about 2.78 s in that pilot. The source-derived copy ledger retains the owned
row-major input, Core's additional column-major transpose, family originals and
device storage where applicable; no resident zero-copy claim is made.

Separate bounded campaigns passed direct-vs-file numeric parity for:

- literal defaults on 256 x 8 across all formats, mixed precision, and four
  cache states (144 trials, three repeats);
- all four metrics, three permutations and two repeats on 4096 x 16 Parquet,
  all profiles and four cache states (144 trials, three repeats);
- time-series and decision-path families on 256 x 8, all formats/profiles
  (72 trials, one repeat; not a repeated timing estimate);
- Float32-source 100k x 100 Parquet mixed across four cache states (48 trials,
  three repeats): observations distinguish new residency, actual native-handle
  reuse, target replacement and cache-disabled execution;
- a larger 1M x 20 Parquet mixed ingestion-isolation case (four trials, one
  repeat): RC2 file call 6.35 s / 1246.8 MiB; candidate 244.7 ms / 576.0 MiB.
  That single larger sample is exploratory, not a robust latency estimate.

The initial family campaign failed in the **collector**, on compact typed arrays
in decision-path result metadata, for both versions. Those raw failures remain
retained. After adding typed-array snapshot support and a focused regression
test, the affected campaign passed. No product change or exclusion was used to
hide that failure. Installed independent NumPy and family-reference gates, the
full Core precision/lifecycle gate (including 27 full-config file cases), and
the Python suite pass separately; route equality alone does not prove those.

Raw specifications, stdout/stderr, per-trial JSON and summary reports remain in
`gafime-rc3-e2e.3DUE8o`. Reference report SHA-256:
`1983e1806a09995d432cb63054297a6aad09a4660ca0017cca6fe1a59171a324`.
Its collector SHA-256 is
`2abc392c831c5c90746daac666ba23578932504bc29db9472ef09b006f90a51e`;
later collector documentation fields do not retroactively alter those samples.

## Permanent coarse cost tripwire

`ingest_cost_budget.json` covers all 18 reference cells above. It deliberately
allows startup/parser/host variation while rejecting the reproduced Python
object amplification, not promising attractive benchmark numbers:

```text
median file wall <= 250 ms + 4 * same-host median direct wall
max file peak RSS <= max(file import peak, max direct peak)
                     + 20 MiB + 12 bytes * rows * columns
```

The committed policy requires at least three matched samples and a declared
source SHA. For this uncommitted development comparison only, a scratch policy
explicitly permitted a null source SHA: the candidate passed all 18 cells;
public RC2 failed both latency and memory limits in all 18 negative-control
cells. This is a regression tripwire, not frozen provenance or release readiness.
Its validator rejects incomplete coverage, missing repeats, mismatched artifact/
fixture/config identities, numerical inequality and unavailable fresh-process
RSS. Synthetic rejection tests exercise those failures independently.

V1 CI builds and installs a specific wheel, runs the same reference collector
and reviewed budget, exercises a small literal-default workflow separately,
and retains evidence even on failure. Hosted success, exact RC3 version/source
identity, independent numerical gates and applicable accelerator/artifact
qualification are still required before RC3 readiness is asserted. A failed
budget is investigated; it is not fixed by silently raising the limits or
substituting resident throughput for public-workflow timing.
