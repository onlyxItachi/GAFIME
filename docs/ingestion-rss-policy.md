# Installed public ingestion cost gates

The gate measures a public file workflow, not only a resident Core executor.
Process RSS includes parser/runtime pages, native storage and retained allocator
pages; it is not a live-buffer counter. Numerical, overflow, ownership, backend,
latency and OOM/process gates remain independent. Arrow is not resident zero-copy.

## Bounded parser-aware model

`ingest_cost_budget.json` selects `independent-parser-v1`. All existing numeric
allowances remain unchanged:

```text
N = rows × feature columns
B = max(file import peak, matched direct-execution peak)
A = existing 20 MiB + existing 12 bytes × N
old memory ceiling = B + A
wall ceiling = existing 250 ms + existing 4 × median direct wall

P = maximum independent parser/projection peak across fresh processes
require P <= old memory ceiling
require median parser wall <= wall ceiling

public memory ceiling = max(B, P) + A
require public peak <= public memory ceiling
require median public wall <= unchanged wall ceiling
```

A failed producer never raises the public ceiling. Its cap is independent of P,
and the revised public ceiling is bounded by `B + 2*A`. Upstream parser growth
cannot grant itself an unlimited allowance. These are catastrophic-regression
guards, not a universal promise or proof of negligible parser overhead.

Direct NumPy execution never exercises the file decoder's producer/runtime floor.
The model accounts for that missing baseline without adding another native buffer
allowance, changing layout, retaining foreign owners, or trimming production
allocators. Required row-major/column-major overlap remains measured.

## Independent controls and provenance

`tests/release_measure/perf_14_parser_control.py` runs isolated fresh processes
with the same authenticated GAFIME wheel/native and import/config declaration
scope as public workers. It calls public Polars CSV/Parquet/IPC readers and literal
name-list projection directly: no product loader, native acquisition, planning or
execution. Polars import/read/projection is timed; frames remain alive until RSS
is frozen. Binary fingerprinting and environment probes happen afterward.

All six narrow/wide × format controls need at least three samples. A Float64
producer may serve all three profiles only because it performs no precision
conversion and has identical source/projection. RC2 needs its own controls, not
a borrowed candidate baseline. Validation matches artifact/source, dependency
versions/native fingerprints, fixture hashes/dtype, reader options, Python,
host/boot/Actions identity, affinity, inherited allocator/thread policy and THP.
The original public `environment` fields remain unchanged. The stable receipt
adds all inherited allocator settings and all four diagnostic THP settings;
separate post-measurement records preserve effective allocator settings, actual
Polars pool size, CPU/memory snapshots and full kernel identity. Volatile free
memory/load/frequency snapshots are retained but are not falsely required to
match. Direct calls need not initialize Polars solely for a provenance probe.
Version fields moved into the shared dependency receipt without being removed;
it additionally binds installed NumPy and Polars native binaries. Complete
receipts are required, not merely equality between two incomplete records.
The CLI verifies the producer's recorded public-report SHA-256 and byte size
against the actual supplied report. Original runner paths remain provenance,
not a requirement to recreate that filesystem when auditing downloaded evidence.
Missing, failed, duplicate, instrumented, treated or mismatched evidence fails
closed. Trimming, sleeps, forced GC and allocator overrides are not qualification.

## Historical verdicts and reproduction

Historical budgets without the model marker, or with `direct-v1`, keep their
original formula. Failed records are not rewritten or retroactively passed.
Current verdicts retain old/new ceilings, producer measurements/caps, public
measurements, failure names and raw input hashes. The original observations stay
in [the matched ingestion evidence](evidence/rc3-ingestion-baseline.md).

After collecting authenticated public direct/file evidence, run:

```bash
python tests/release_measure/perf_14_parser_control.py \
  --public-report /scratch/reference/report.json \
  --repeats 3 --output /scratch/parser
python tests/release_measure/perf_14_ingest_budget.py \
  --report /scratch/reference/report.json \
  --parser-report /scratch/parser/report.json \
  --budget tests/release_measure/ingest_cost_budget.json \
  --variant candidate --output /scratch/cost-verdict.json
```

Use new output paths. V1 CI preserves both collectors even on failure. Cost
success alone never sets release readiness or substitutes for independent
numerical, artifact, accelerator or release qualification. Allocator attribution
and Polars version comparisons are separate investigations, not dependency
downgrade instructions. A version difference alone does not prove a defect.
The [bounded Polars comparison](evidence/rc3-polars-rss-comparison.md) records
the version-dependent observations and why no dependency downgrade was applied.
