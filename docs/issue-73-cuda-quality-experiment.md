# Issue #73: candidate quality and CUDA reuse experiment

This extends the [first Core feasibility checkpoint](issue-73-native-evidence-feasibility.md)
on the same draft branch. It is **not a supported public API**, a release, full
#73 completion, or proof that all future #72 requirements are unnecessary.
The maintainer authorized local quality/backend/performance investigation after
that checkpoint; main and the separate RC2 hardening lane remain untouched.

## What is being tested

- Native Rust materialization of identity, absolute difference, softsign, and
  the existing centered-product reference form into a deterministic candidate bank.
- Same-bank mixed Pearson scoring through the existing production Core and
  CUDA executors, not a replacement serial Core leaf benchmark.
- Training-only paired-view selection followed by untouched holdout scoring in
  planted fixtures. Synthetic positive/negative controls test evidence mechanics;
  they cannot establish downstream DL Method, TabFM, or general feature-quality gains.
- A bounded row/candidate-count sweep that separates native materialization,
  allocation/upload/planning, and synchronous resident scoring.

## Placement and non-claims

| Work | Placement in this experiment |
| --- | --- |
| Input validation, candidate catalog, materialization, selection | Rust / Core |
| Same-candidate paired-view evidence | Core mixed native statistics |
| Common anchor or actual held-out label correlation | Explicit Core or CUDA mixed scorer |
| Supplied-graph evidence | Original Core probe only; not added to CUDA |
| CUDA device kernels, launchers and ABI | Existing implementation, unchanged |

An unlabeled anchor is a real observed feature column. It occupies the current
ABI's row-aligned `target` operand; no dummy target or fake labels are introduced.
This is **mathematical reuse of a common-reference scorer**, not a new target-free
GPU protocol. Each paired-view candidate would have a different reference, while
neighbor evidence needs edge buffers. Neither acquires CUDA support through
this adapter. A future native lowering needs an explicit reviewed contract.

New transforms do not execute on the GPU. Native materialization followed by
CUDA scoring follows the generated-family placement already used for time-series
and decision-path features. It makes transfer and bank-storage costs visible;
it does not prove this approach is ideal for arbitrary large feature programs.
Source pairs and materialized columns retain an experiment-side catalog mapping,
not a new public candidate identity or inference-delivery contract.

## Numerical and quality protocol

Inputs and pointwise arithmetic are f32; means/statistics are f64 with f32
centering operands for mixed pointwise products. The independent score oracle
accumulates means and centered products in row order using f64, without calling
GAFIME correlation helpers. The finite experiment checks absolute error at
`1e-12`, the existing Core mixed regrouping bound, and reports measured CUDA
error against that same bound. This does not loosen production tolerances.
No fast-math flags, precision changes, or implicit backend fallback are allowed.

The quality fixtures have disjoint training and holdout rows/RNG domains.
Selection accepts only paired training banks, never the holdout labels. A
shuffled-view negative control is reported alongside the aligned-view scores.
The candidate set and seeds are fixed before observing results. Planted latent
targets deliberately encode the synthetic mechanism: success is a control test,
not an empirical demonstration of generalization on real datasets.

The initial probe's unavailable-evidence semantics remain unchanged. The
production unary scorer retains its existing zero sentinel for constant inputs;
selection must reject unavailable/constant evidence rather than rank that zero
as a desirable quality score. Shapes, descriptor indices, duplicate identities,
finite values, transformed overflow, and bounded allocation sizes are validated.

## Performance protocol and source cost

The native runner uses default Rayon worker selection for the Core executor and
Rust materialization. It validates numerical results before warmup/timing and
checks repeat result identity afterward. Each seeded, shuffled benchmark cell
runs in a fresh process, with 10 warmups and 30 retained samples, targeting at
least 100 ms per measured region with a bounded calibration loop. Measured
region durations are retained; shorter regions must not be presented as meeting
the target. Report raw samples and observed worker counts.

Resident scoring includes the existing executor and result-buffer writes, not
host result extraction, candidate generation, or upload. Materialization, setup,
and preparation plus first execution are separate **single observations**, not
statistically qualified cold/end-to-end speed claims. Inputs, catalog and anchor
already exist. The direct-product control additionally follows oracle-only bank
materialization, which can warm allocator/caches even though it is excluded from
that control's preparation interval. Candidate-row throughput is
not a FLOP/s estimate, achieved occupancy, or proof of hardware saturation.
Common-bank Core/CUDA scores are comparable; different candidate mathematics
and different public families must not be relabeled equal-work benchmarks.

The host is a shared desktop. Pre/post snapshots retain CPU process activity,
GPU utilization/clocks/memory/power, allowed CPU set and AC state. They do not
prove absence of interference between snapshots. Until independent hardware
counters and an uncontaminated run support a stronger statement, retained
timings are **bounded shared-host diagnostics**, not release/performance gates.
No drivers, clock policy, affinity, power settings, or unrelated workloads are changed.

The bank is capped at 64 MiB of f32 values, not 64 MiB total process memory.
Materialization also allocates a candidate-major temporary, worker scratch,
f64 source-column sums and f32 source-column means. Its checked conservative
numeric-payload bound is `3 * bank_bytes + 12 * source_columns`, excluding inputs,
vector/catalog metadata, allocator slack and backend resident copies. This
deliberately overcounts phases that need not overlap; mean storage can dominate
for wide inputs and few candidates. Full-bank materialization costs
`O(rows * candidates)` storage
and upload, unlike the direct continuous centered-product descriptor route.
These costs are part of the feasibility answer, not hidden behind kernel timing.

The source accountant preserves the initial 600-line envelope result separately
and reports expansion, fixtures, wrappers, runner and documentation costs. It
does not reinterpret the original overrun or classify test infrastructure as
zero maintenance cost. Shipping source, public API, ABI, dependencies, versions
and distribution topology remain unchanged.

The expanded native candidate/scorer/harness layer, including fixtures and tests,
is **1,529 nonblank non-line-comment lines / 1,690 physical lines**. The local
Python evidence runner is **224 / 247** respectively. These are additions to the
initial 654-line native probe, not a claim that the CUDA/quality expansion stayed
inside its original 600-line envelope. Retained JSON/CSV samples are accounted
separately as data, not implementation. Reuse avoided shipping engine duplication;
it did not make the validation and adapter maintenance cost negligible.

## Reproduction

Build the unchanged product libraries and a standard RT-disabled CUDA payload:

```bash
CARGO_TARGET_DIR=target/issue73-native cargo +1.97.1 build --release --locked \
  -p gafime-cpu -p gafime-gpu-sys -p gafime-orchestrator -p gafime-types -j 2
cmake -S src/cuda -B target/issue73-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGAFIME_CUDA_ARCHITECTURES=89-real \
  -DGAFIME_CUDA_RT_BUILD_MODE=off -DGAFIME_CUDA_BUILD_TESTS=ON
cmake --build target/issue73-cuda --parallel 1
ctest --test-dir target/issue73-cuda --output-on-failure
python tests/release_measure/run_issue73_quality_cuda.py \
  --cargo-target target/issue73-native \
  --cuda-lib target/issue73-cuda/libgafime_cuda_v1.so \
  --output target/issue73-evidence
```

The shown architecture is for the measured SM89 device, not a replacement for
the release matrix. Use an appropriate supported local target on other hardware.
Add `--timings` only for the bounded diagnostic sweep. The output directory must
not exist: previous evidence is never silently overwritten. The report retains
source state, binary/rlib/payload SHA-256 hashes, commands, controls and raw cells.
It is not frozen-wheel release provenance.

## Results: 2026-09-06 local checkpoint

The completed 54-cell sweep ran from clean commit
`97e166bceb31950311508e89b6df534b87870ec2`; source identity was unchanged at its
end. [Retained raw timing/quality extract](evidence/issue73-core-cuda-2026-09-06.json)
contains all 1,620 resident samples, measured-region durations, per-cell hashes,
quality controls, source/build identities and host snapshots. Full score vectors
and per-cell CPU process lists remain in the hashed local original report at
`target/issue73-cuda-sweep-97e166b/report.json`; the extract records their computed
cross-backend and equal-work comparisons. These are local experiment artifacts,
not frozen-release evidence. Product source was unchanged from the base; the
payload and rlibs were built once and reused.

Independent review found two bookkeeping defects after that run: the original
peak field omitted the mean buffers, and `cold_end_to_end` overstated the direct
control's coldness. Raw evidence is retained unchanged with explicit errata.
The follow-up adds checked accounting and a wide-input regression, renames the
preparation field and exposes the prior oracle materialization, and rejects
source/payload/harness identity drift. These corrections do not change candidate
values or resident scoring. The timings describe the measured commit, not a
claim that the later documentation/correction commit was re-benchmarked.

### Feature quality: successful control, not universal utility

Across seeds 73–77, training-only paired-view selection chose absolute difference
in all five planted invariance fixtures. Holdout Pearson was
**0.998727–0.998866**; aligned training evidence was approximately 1, while the
shuffled-view control ranged **-0.047414 to 0.042071**. Raw identity holdout
correlation was **-0.041974 to 0.056799**. This establishes the intended synthetic
invariance mechanism without target-bearing selection.

The linear-latent control supplies the important counterexample. Softsign won
training view consistency in all five cases, but its holdout correlation was
**0.985800–0.988212**, below raw identity's **0.999373–0.999562**. More view
consistency does not imply more information for a later consumer's target.
Selection policy, representation quality and downstream utility must remain
separate contracts. No DL Method or real-dataset efficacy test was performed.
Centering in the product control is batch-local, not a fitted train-statistics
deployment contract; neither selected feature depends on that centering.

All Core/CUDA quality outputs passed the independent f64 oracle (maximum absolute
error **9.9921e-16**). Across the timing cells, maximum oracle error was
**2.9144e-16**, Core/CUDA score difference **1.1103e-16**, and direct versus
materialized product score difference **zero** on each backend. Within-backend
repeat outputs were identical. Precision here is mixed only.

### Equal-work execution and utilization

Host: Ryzen AI 9 HX 370, default **24 Rayon workers**, RTX 4060 Laptop 8 GiB
(SM89), CUDA 13.3.73, driver 610.57.04, GCC 15.2 and Rust 1.97.1. Each backend ran
sequentially on the shared desktop. At **32,768 rows × 253 candidates**, median
resident execution was:

| Candidate execution path | Core | CUDA |
| --- | ---: | ---: |
| Materialized absolute difference → unary scorer | 466.807 µs | 419.758 µs |
| Materialized centered product → unary scorer | 545.233 µs | 419.536 µs |
| Existing direct centered-product executor | 556.119 µs | 948.474 µs |

Only the two product rows are mathematically equal-work controls. Absolute
difference has different feature semantics. Materialized unary scoring follows
the same broad placement as generated time-series/decision-path families, but
this is **not** an end-to-end benchmark of those public families or a comparison
against ROCm/Metal. Production installed-wheel correctness smokes separately
passed continuous, time-series and decision-path execution on Core and CUDA
for the supported four metrics.

The materialized CUDA product score is about 2.26× faster than direct product
**in this resident-only cell**. It buys that by moving pointwise work out of the
timed executor and storing a 31.625 MiB bank instead of executing from the
2.875 MiB source matrix. Generation alone took approximately 22–25 ms in the
largest-shape single observations; allocation/upload/planning adds further cost.
This is a reuse/amortization trade-off, not a one-shot or universal speedup.
For the smallest shape (512 × 15), Core's materialized product median was
12.131 µs versus CUDA's 13.303 µs. CUDA is not uniformly preferable.

At 32,768 rows, CUDA absolute-difference resident throughput rose from 9.865 to
16.919 to 19.750 billion candidate-rows/s for 15, 66 and 253 candidates. This
shows improving utilization with diminishing gains within the sampled range,
not a proven asymptotic saturation point. 197 of 1,620 regions were below the
100 ms calibration target (minimum 79.215 ms); all raw samples are retained.

Three separate, single-kernel Nsight Compute captures used the same measured
harness/payload, with clocks/cache policy unchanged. They are profiling
observations, not additional unprofiled timing samples:

| Largest-shape kernel | SM throughput | Achieved occupancy | DRAM throughput |
| --- | ---: | ---: | ---: |
| [Absolute-difference bank unary scorer](evidence/issue73-ncu-absdiff-2026-09-06.csv) | 83.02% | 80.56% | 17.51% |
| [Product bank unary scorer](evidence/issue73-ncu-product-2026-09-06.csv) | 83.01% | 80.52% | 17.25% |
| [Direct centered product](evidence/issue73-ncu-product_direct-2026-09-06.csv) | 78.62% | 83.66% | 1.01% |

These are the profiler's named counters for the selected scoring kernel, not
percentages of whole-pipeline peak FLOP/s. They show substantial physical GPU
use and no hidden Core substitution; neither occupancy nor this three-point
sweep proves full HPC saturation. The profiler warns about uncontrolled caches
and unmodified clocks. CPU generation, transfers, other kernels and desktop
activity remain outside those individual-kernel percentages.

### Architecture conclusion

**Yes**, these bounded candidate types can reuse existing Core/CUDA scoring
without duplicating an engine, modifying the ABI or adding a Python data plane.
**No**, that does not establish a general GPU-native target-free evidence engine
or remove all of #72's motivation. Paired references, graph inputs, candidate
program identity and inference delivery still need explicit product contracts.
Full-bank cost and selection-policy ambiguity are the next concrete design
constraints; neither warrants inventing the complete IR just to run this probe.

Local validation also passed 10 physical CUDA ABI/lifecycle fixtures (including
ABI 1.0, 1.1 and synthetic future 1.2), the installed-wheel public family smokes,
release/MSRV probe tests and strict Clippy, formatting, V1 architecture, policy
and source-tree composition checks. Exact-head hosted status and independent
AI Review Record are recorded on draft PR #92. This checkpoint is not merge or
release approval; #72/#73 remain open and main/RC2 are unchanged.
