# Local RT compact conditional-evidence experiment

> **Planned local experiment; implementation and physical validation are in
> progress.** This document does not claim a completed build, device execution,
> speedup, supported backend, package capability, a new public Python API, or
> a standard payload ABI. It is a bounded follow-up to
> [local RT semantic batch materialization](rt-semantic-batch-experiment.md),
> not a replacement for that cold dense-column baseline or for the historical
> [decision-path score experiment](rt-gbdt-cuda.md).

## Question and non-goals

Can a finite frozen-region batch answer a small set of **conditional evidence
questions** with compact exact counts before its region columns are
materialized? The useful questions are deliberately narrow:

- target-free per-rule occupancy;
- binary same-program agreement across a declared paired view; and
- an optional, partially labeled binary split-Gini question.

This is not a target protocol, a tree inducer, a Gini feature family, a
universal unsupervised/self-supervised quality score, or evidence that RT cores
are generally faster. It does not broaden the predicate language, create a
generic candidate catalog in CUDA, alter evidence identity/selection policy,
advertise ordinary CUDA capability, change `auto` placement, or authorize a
release artifact. CUDA RT/OptiX stays an explicit local CMake experiment whose
sources, PTX, libraries, reports, and benchmark output remain outside wheels,
sdists, workflow/cache artifacts, frozen bundles, and GitHub Releases.

The source tree does add Rust semantic vocabulary and an explicit local-CMake
compact-query ABI for this experiment. Those are deliberately not a new public
Python API or standard semantic payload ABI.

The accepted mathematical program remains Rust-owned. CUDA sees only checked
physical slots and frozen fp32 region descriptors. During query creation and
compact execution it receives neither `FeatureId`/candidate IDs, evidence
names, provenance, selected candidates, a generic target field, nor an output
slot into which it could materialize a dense root column. A separate,
post-evidence selected-coverage operation may later name one fresh physical
slot; it is described below and is not part of the compact evidence record.

The corresponding semantic evidence variants are deliberately restricted to
canonical `HardPredicate` and `DecisionRegion` candidates:
`BinaryOccupancy`, `BinaryPaired { Agreement | IntersectionOverUnion, view }`,
and `BinaryLabeledGiniGain { labels }`. Separately, `RegionCount { regions }`
is a bounded canonical, target-free candidate form that sums distinct accepted
decision-region memberships. It is a real new mathematical feature, not an
evidence alias, but it does not broaden the predicate language or authorize
tree discovery.

## Planned compact lifecycle

The default optional executor seam is:

```rust
NativeEvidenceExecutor::evaluate_compact(
    registry, frame, candidates, channels, retained, max_bytes,
) -> SemanticResult<Option<CompactEvidenceBatch {
    dependencies, channel_values, explicit_peak_bytes,
}>>
```

`Option` records whether this executor can answer the complete request by its
compact route; `SemanticResult` carries validation, admission, and native
failure. `Option` is not a partial-success signal. Rust remains the owner of
whether to issue this optional local query and which explicit route to request.
The new query itself has no implicit fallback: an explicit local-RT request
must fail closed rather than let native code substitute another backend or
silently change the query.

The lifecycle is intentionally different from the earlier dense materializer:

1. Rust validates frame/profile/context alignment, program dependencies,
   resource bounds, and channel declarations. Non-region dependencies such as
   an accepted softsign atom may be resident, but candidate **region roots are
   not materialized before evidence**.
2. Rust resolves canonical programs to physical frozen-region terms plus region
   and partition offsets, and holds the relevant immutable fp32 semantic-bank
   `Arc`s for the query lifetime. When a paired bank has a different retained
   slot layout, it supplies one physical paired-input slot per term rather than
   copying either bank. It retains dependency metadata in the returned batch;
   that metadata is not a second native candidate catalog.
3. The local query copies those physical descriptors at creation. It owns its
   point packing, geometry, IAS/GAS/SBT, workspace, and query-local execution
   state. Labels are copied for each execute call; no label pointer or caller
   descriptor pointer is cached.
4. One execute asks for all declared compact channels. It returns either a
   complete per-region record set and states, or an error. A bad paired view,
   label vector, budget, geometry shape, or native failure cannot leave
   occupancy values looking like a partially completed evidence table.
5. Rust maps the returned count records to the existing named evidence channels
   and performs ordinary policy/acceptance. Only after `accept_with` chooses
   candidates are those selected region columns materialized into the semantic
   bank for later reuse or Arrow delivery.

This preserves the distinction between a compact evidence query and a dense
feature value. It also keeps overlapping regions legal: no first matching
region is treated as an exclusive leaf assignment.

## Exact compact statistics

For region `r`, row `i`, and frame `F`, let `m^F[r,i]` be the exact binary
result of the canonical conjunction. The local envelope admits finite,
non-subnormal fp32 input values and thresholds and preserves `<=` and `>`
semantics, including signed zero. OptiX traversal is only culling; the exact
guard determines `m`.

### Target-free occupancy/rule records

Rust retains the structural `term_count[r]` from the frozen descriptor range
alongside each returned record. The native compact record itself carries exact
count

```text
inside[r] = sum_i m^F[r,i]
outside[r] = rows - inside[r]
```

as checked `u64` integers. `term_count` supplies rule complexity context; it
is not a claim that overlapping rules define one exclusive assignment.
`BinaryOccupancy` may request the local fp32 device finalizer `inside / rows`,
which Rust maps only after it has checked the exact returned count; other
profiles retain their normal semantic finalization path. There is no fabricated
target, density objective, or universal USL score.

### Selected target-free coverage feature

Coverage is a distinct, explicitly selected **feature** rather than another
binary evidence channel. After a complete compact execute has established
query-local primary membership state, the local coverage endpoint may write one
fresh primary-bank slot with

```text
coverage[i] = sum_r m^F[r, i]
```

for the submitted region set. Thus a row can receive `0` through
`region_count`, including values greater than one for legal overlap. It is not
a target, a weighted score, a generic expression, a per-region dense root, or
a replacement for `BinaryOccupancy`'s one-record-per-rule count. Before a
successful execute there is no retained membership and the endpoint rejects;
after success it commits only its named fresh slot after synchronization. The
implementation may retain overlap-safe masks or use a proven direct-first-hit
per-row count internally; neither representation changes the exact result.
Rust still decides whether a selected `RegionCount` candidate may materialize
after `accept_with`; the compact query does not prepopulate it merely because
the bank has spare physical capacity. The native local lowering admits two
through 64 distinct canonical decision regions, while the registry-effective
envelope is `2..=min(64, ProgramLimits.max_logical_arity)`: its default
`max_logical_arity` is eight, and an explicit Rust registry-limit increase is
required for nine through 64. A single binary region is not duplicated under a
second candidate identity; this does not change product defaults.

### Paired binary agreement

For two explicitly aligned frames `A` and `B`, the query returns the exact
per-region contingency tuple in this fixed order:

```text
n00 = count(m^A = 0 and m^B = 0)
n01 = count(m^A = 0 and m^B = 1)
n10 = count(m^A = 1 and m^B = 0)
n11 = count(m^A = 1 and m^B = 1)
```

Thus `n00 + n01 + n10 + n11 == rows`, primary/paired occupancy are
`n10 + n11` and `n01 + n11`, and binary agreement is
`(n00 + n11) / rows`. Optional IoU is `n11 / (n01 + n10 + n11)` and is
`ConstantOperand` when its union is zero. This is per-region binary membership
agreement, not an invented first-hit leaf assignment; each overlapping region
gets its own tuple. These are respectively the `BinaryPaired::Agreement` and
`BinaryPaired::IntersectionOverUnion` channel meanings.

Agreement alone can favor a trivial region that is constantly outside (or
inside) in both views. It is therefore paired conditional evidence, not a
universal SSL-quality objective. A profile can require nontrivial occupancy or
combine its separately selected coverage policy with occupancy constraints, but
the compact query does not silently turn those checks into a weighted score.

### Partial binary labels and Gini

Labels are an optional row-keyed subset. Rust validates that every supplied
label is exactly `0` or `1`; unlabeled rows have no implicit value and do not
enter label statistics. For a primary frame, the record contains exact `u64`
`label_support` plus counts in this fixed order:

```text
outside_label0, outside_label1, inside_label0, inside_label1
```

Let `support` be their sum, `N_b` the total for branch `b in {outside, inside}`
and `N_{b,y}` its class count. The optional fp32 finalizer uses:

```text
gini(b) = 1 - (N_b0 / N_b)^2 - (N_b1 / N_b)^2
parent  = 1 - (N_0 / support)^2 - (N_1 / support)^2
gain    = (parent - (outside / support) * gini(outside))
          - (inside / support) * gini(inside)
```

It does not clamp the result. `support < 2` is `InsufficientSupport`; an empty
inside/outside branch or a globally single-class labeled subset is
`ConstantOperand`. An absent label binding is `MissingLabels`, not a numerical
zero. The exact-count tuple is the correctness authority. The generic semantic
channel finalizes in f32 for fp32 and in f64 for mixed/fp64; the local OptiX
query itself is fp32-only, so its optional device finalizer is f32. Every
finalizer is checked only after exact count equality and reports its state
separately.

A hybrid request simply asks for several of these named channels together.
Its policy keeps occupancy, paired agreement/IoU, and binary-label Gini in
their own units and applies explicit constraints/Pareto/primary ordering. It
must not synthesize a weighted universal score.

## Native ownership, admission, and resource rules

The local CUDA surface is an opaque create/execute/free query API in the RT
headers and sources only. Creation copies immutable physical
`GafimeSemanticFrozenRegionTerm` descriptors and region/partition offsets from
the fp32 semantic bank(s). A partition is a contiguous range of submitted
region ordinals, not a candidate-id range and not a proof of non-overlap.
When a paired bank exists, a same-length paired-slot array maps every term to
that bank's layout. Execution accepts channel flags and contextual physical
inputs, then emits fixed per-region records plus optional fp32 finalizers/state
fields. An unrequested count channel is zero and finalizer state zero means the
finalizer was not requested. No target/f64 reduction or dense membership output
is part of the compact route.

The reusable compact query has its own bounded envelope: at most 262,144 rows,
8,192 regions, 64 terms per region, three physical axes per region, and 64
logical region groups. These are not the earlier cold dense-materialization
limits of 65,536 rows and 256 regions. A query must select exactly one route:
OptiX RT (`REQUIRE_RT`), the query-binned CUDA comparator (`FORCE_SM`), or the
diagnostic exact all-region SM comparator (`FORCE_SM_EXHAUSTIVE`). Zero route
flags and every multiple-selector combination are `INVALID_ARGUMENT`. The new
compact-query API has no default route and never falls back from RT to SM after
admission; Rust chooses a route before creating it. This does not alter legacy
RT API policy.

The query reports separate persistent and temporary explicit-buffer needs.
Persistent accounting includes query-owned geometry, packed points,
acceleration structures, SBT, copied descriptors, and fixed records; temporary
accounting includes overlap-safe traversal state and execute-local label copy.
Both must be admitted before the relevant allocation/launch. Opaque driver and
OptiX overhead is still outside a process-RSS guarantee. Query teardown follows
the existing per-device ownership, serialization, and caller-device restoration
rules.

The normal path is duplicate-safe for overlapping regions: repeated traversal
callbacks may not increment any count twice. First-hit is an internal native
optimization, never a caller geometry selector. It is permitted only after
native proof that every resolved group is finite, bounded, 2D, and pairwise
non-overlapping; otherwise the overlap-safe mask path remains in force. Tests
therefore prove equal exact records for overlapping and partition-shaped cases,
not a guessed internal optimization branch.

## Independent oracle and planned validation matrix

The correctness oracle is a host-side direct evaluator that does **not** call
the RT query, its geometry planner, or a dense materialization result. For each
row and region it evaluates every frozen term with native fp32 `<=`/`>`
comparison, ANDs the terms, and increments checked `u64` occupancy, paired
`n00/n01/n10/n11`, and partial-label contingency bins. It uses the same
declared row-key alignment but independently reconstructs all count tuples.
It must compare count records exactly before checking any fp32 finalizer state
or value.

The planned fixtures are intentionally small enough to diagnose a wrong bin,
predicate, or duplicate guard, then broad enough to exercise the native batch
limits:

| Fixture | Required observation |
| --- | --- |
| Tiny exact table | 8--16 rows; one-, two-, and three-term regions; signed-zero and strict/open threshold boundaries; overlapping rules; all paired contingency cells populated. Exact record order and every `u64` field match the independent oracle. |
| Target-free occupancy | Mixed occupancy from empty, sparse, dense, and overlapping regions. Check `inside + outside == rows`, Rust-retained static term counts, fixed candidate order, and no dense root slot initialized before evidence. |
| Selected coverage feature | After a complete compact execute only, materialize one fresh primary-bank slot and compare its per-row integer-valued float output with `sum_r m[r,i]`, including legal overlap. Reject before execute and reject an already initialized output slot. This is distinct from the `O(regions)` binary evidence record table. |
| Paired SSL | Same frozen programs on aligned two-bank rows with deliberate membership flips. Verify all four `n00/n01/n10/n11` cells, agreement/IoU states, equal-frame agreement, and rejection of row-key/domain/row-count misalignment. |
| Partial labels | Binary labels on a strict subset, including unlabeled rows that would alter an answer if treated as zero. Verify exact outside/inside-by-label counts, `InsufficientSupport`, `ConstantOperand`, a measured zero, and a nonzero Gini gain. |
| Hybrid atomicity | Request occupancy, paired, and labels together. Verify every channel against the oracle, then make one context invalid and prove no partial compact batch or evidence record escapes. |
| Bounds and lifecycle | Reject mixed/fp64, nonfinite/subnormal values, over-64-term regions, over-three-axis regions, malformed offsets, under-persistent/temporary budget, and stale/destroyed resources before output. Confirm copied descriptors survive caller-vector mutation, labels are recopied per execute, bank lifetimes are retained, and final owner teardown restores the caller CUDA device. |
| Overlap/first-hit | Use overlapping boxes whose callback counts would overcount without the duplicate guard, plus a finite bounded non-overlap partition. Compare each exact record with RT and forced-SM routes; do not infer an internal first-hit branch from a caller selector. |
| Triangle-eligible 2D partition | Use finite, conservatively wide, pairwise non-overlapping 2D rectangles. Compare exact records and fp32 finalizer bits across RequireRT, exhaustive SM, and query-binned CUDA without asserting the internal triangle choice. |
| Multiple axis groups | Submit contiguous partition ranges with different physical primary/paired axis pairs, each bounded and 2D. Verify grouped RT preserves submitted result ordinals and matches exhaustive/query-binned exact records; direct first-hit remains a single-group-only optimization. |
| Limit shape | Exercise the compact query's own row/region/group/term limits (separately from the old dense 65,536/256 envelope) using compact outputs; assert output is `O(regions)` records, not a hidden `rows * regions` membership delivery. |

### Canonical tiny oracle fixture

The first direct-ABI and Rust fixture should share this intentionally
non-partitioned eight-row case. `x_A` and `x_B` are the two aligned views;
`?` is an absent label. Both views use the same `y` column:

| row key | `x_A` | `x_B` | `y` | label |
| --- | ---: | ---: | ---: | --- |
| 0 | -1 | -1 | 0 | 0 |
| 1 | -0 | 1 | 1 | 0 |
| 2 | +0 | +0 | 0 | ? |
| 3 | 0.5 | 0.5 | 0 | 1 |
| 4 | 1 | -1 | 1 | 1 |
| 5 | 2 | 2 | 1 | 1 |
| 6 | -2 | 2 | 0 | ? |
| 7 | 3 | 3 | 2 | 1 |

The three regions, in returned order, are:

```text
r0: x > -0
r1: x > -1 AND x <= 2 AND y <= 1
r2: x > -2 AND x <= 1
```

The oracle records must be exactly:

| region | terms | occupancy `(inside,outside)` | paired `(n00,n01,n10,n11)` | labels `(out0,out1,in0,in1)` |
| --- | ---: | --- | --- | --- |
| `r0` | 1 | `(4,4)` | `(2,2,1,3)` | `(2,0,0,4)` |
| `r1` | 3 | `(5,3)` | `(2,1,1,4)` | `(1,1,1,3)` |
| `r2` | 2 | `(5,3)` | `(3,0,0,5)` | `(0,2,2,2)` |

This one case catches strict/open signed-zero handling, all paired contingency
cells, legal region overlap, a pure split, mixed split branches, absent labels,
and a defined IoU path. It is a count oracle, not a performance shape.
Additional tiny fixtures must cover an empty direct-ABI interval (all-zero
membership if admitted by the shared geometry builder), under-support, empty
branch, global-single-class, and union-zero states.

Direct-C-ABI coverage will live in a local-only
`tests/gpu/cuda_rt_semantic_compact_smoke.cpp` CTest fixture, with return code
77 reserved for absent RT hardware. The Rust local-CMake integration test will
cover the semantic lifecycle: no dense roots before evidence, selection first,
then selected-only materialization/reuse. Timing code remains ignored and
outside ordinary CTest/release gates.

## Fair comparison plan

Correctness and speed are separate evidence. The host integer oracle is a
correctness oracle, not a GPU performance baseline. A compact-query timing
record must identify exactly which of these comparable lanes ran:

1. **Exact CUDA SM exhaustive compact comparator.** It consumes the same physical
   descriptors, channels, contexts, output records, and budget policy, but
   evaluates every submitted conjunction without OptiX. Its exhaustive region loop is a valid
   implementation baseline and establishes exact same-statistic parity; it is
   not algorithmically matched for partition-shaped inputs. The local-only
   `FORCE_SM_EXHAUSTIVE` selector is the explicit diagnostic hook; it is not a
   public execution policy.
2. **Query-binned CUDA comparator.** For finite 1D--3D frozen boxes, it builds
   an exact ordered-float bin lookup, maps each row to candidate regions, and
   guards every candidate with the original open/closed fp32 predicates. It
   supports legal overlap through per-cell region lists and includes its cold
   build cost. In this bounded pass it still writes overlap-safe membership
   masks and uses the shared statistics reducer; it is therefore an exact,
   useful query comparator, not a claim of the best possible CUDA partition
   algorithm. The local `FORCE_SM` query flag exposes it for parity and
   benchmark orchestration; it is not a public geometry selector.
3. **Proof-gated RT direct accumulation.** When native proof establishes
   exactly one finite, bounded, 2D pairwise-non-overlapping group, RT may
   accumulate the compact records directly without an `regions × rows`
   membership mask. Otherwise it uses the overlap-safe representation. A native
   CUDA direct partition-accumulation comparator is future work and must not
   be inferred from the present query-binned lane.
4. **Dense materialize then reduce.** This remains useful to measure the
   delivery/memory consequence of the old route, but it produces and transfers
   `rows * regions` memberships. It is not the same compact-output workload and
   must be labeled accordingly.

Each candidate comparison uses identical programs, source/accepted dependency
state, paired rows, label mask, channels, count-record cardinality, precision,
device, compiler/runtime, power/display state, worker policy, and explicit
budget. Report distinct phases: shared bank/dependency preparation,
query/index/geometry construction, compact execution, and compact host
delivery. A cold end-to-end lane may include source upload and call-local
geometry/index construction; a resident query-only lane starts only after the
same bank/dependencies are ready. Neither may be relabeled as the other, and a
future generic-program cache is a separate category from fresh execution.

The historical `118.077x` first-hit compact-score ratio at `65,536 x 8,192`
compared RT against an exhaustive legacy SM fallback with a resident target. It
does not measure this target-free/paired/partial-label compact query and is not
a matched partition-index comparison. It must not be cited as a baseline,
denominator, expected speedup, or result for this experiment.

After exact correctness on the configured device, the opt-in
`gafime_cuda_rt_semantic_compact_smoke --benchmark` diagnostic may use exactly
two finite single-group 2D grids: `65,536` rows by `1,024` regions and
`262,144` rows by `4,096` regions. Both request paired and partial-binary-label
channels plus every finalizer. It first verifies all returned `u64` records,
fp32 result bits, and finalizer states agree across RequireRT,
`FORCE_SM_EXHAUSTIVE`, and the query-binned CUDA `FORCE_SM` lane; only then does it time three warmups and
nine rotated raw calls **per lane**. It reports cold query creation from
resident inputs separately from warm synchronous execute, prints each raw
nanosecond sample and median, and never prints a ratio or speedup. The first
implementation uses one group and caps each query at 300 MiB persistent and
64 MiB temporary explicit allocation; it records the live-query memory
snapshot and neither source upload nor teardown belongs to either timed phase.
This diagnostic measures compact evidence queries only; it does not time
selected `RegionCount`/coverage materialization.
The current RTX 4060 Laptop shared desktop state is acceptable for a clearly
labeled bounded diagnostic after correctness; it is not a pristine-lab
comparative claim. Older driver-610 measurements are historical and cannot
qualify a new driver-615 result.

## Reproduction status

The existing local build directory has the OptiX headers and
`/usr/local/cuda/bin/nvcc`, but its live cache currently records architecture
`75` and an empty `CMAKE_BUILD_TYPE`. A retained measurement must explicitly
reconfigure the intended `89-real;89-virtual` Release build and record the
resulting cache and hashes. Planned correctness commands, once the source and
header land, are:

```bash
cmake -S src/cuda -B target/rt-batch-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DGAFIME_CUDA_BUILD_TESTS=ON \
  -DGAFIME_CUDA_RT_BUILD_MODE=on \
  -DGAFIME_OPTIX_INCLUDE_DIR=/home/hamza-usta/SDKs/optix-sdk/include \
  '-DCMAKE_CUDA_ARCHITECTURES=89-real;89-virtual'
cmake --build target/rt-batch-native --parallel 1 \
  --target gafime_cuda_rt_semantic_compact_smoke
ctest --test-dir target/rt-batch-native \
  -R gafime_cuda_rt_semantic_compact_smoke --output-on-failure
GAFIME_CUDA_V1_LIB="$PWD/target/rt-batch-native/libgafime_cuda_v1.so" \
  cargo +1.97.1 test -p gafime-gpu-sys --locked \
  --features local-cmake-experiment \
  --test local_cmake_experiment_numeric_domain \
  local_semantic_compact -- --nocapture
```

The optional benchmark command is not a CTest and must follow a passing
correctness CTest on the exact same build:

```bash
target/rt-batch-native/gafime_cuda_rt_semantic_compact_smoke --benchmark
```

Its bounded stdout records are accepted only as device-specific experiment
evidence after the preceding exactness gate; they are not a general performance
claim, release artifact, or substitute for a matched future comparison.

### Initial physical-correctness checkpoint (2026-09-11)

The first local checkpoint was built from the uncommitted experiment worktree
whose base commit was `17aef1e2a2f68ee20d3d7edc12866c1442809eef`, with
`CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=89-real;89-virtual`,
local OptiX headers, and the compact CTest target only. On an NVIDIA GeForce
RTX 4060 Laptop GPU with driver `615.71.09`,
`gafime_cuda_rt_semantic_compact_smoke` passed `1/1` in `0.77 s`. The test
exercised strict RT plus both exact SM comparator modes against its direct
integer oracle, including occupancy, paired and partial-label records,
finalizers, copied descriptors/labels, coverage lifecycle, budgets, overlap,
and caller-device restoration.

This is a bounded physical correctness result for the then-current
single-group Custom-AABB path, not a grouped-IAS/GAS or triangle-reuse
qualification. No `--benchmark` invocation, timing evidence, ratio, speedup,
package artifact, or release claim was produced. The current device had
`612 MiB` reported in use and `31%` utilization when the post-test snapshot was
taken, so even a future bounded diagnostic must retain the shared-desktop
qualification. Grouped geometry and any later physical result require a fresh
build and CTest at their exact source snapshot.

The final test names and command suffixes remain provisional until the local
header/FFI implementation lands. No GPU job was run while writing this design.
Any completed compilation, source inspection, CTest registration, physical
correctness, and timing evidence will be recorded separately against its exact
head and artifacts.
