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
host result extraction, candidate generation, or upload. Cold materialization,
setup, and combined first execution are separate **single observations**, not
statistically qualified end-to-end speed claims. Candidate-row throughput is
not a FLOP/s estimate, achieved occupancy, or proof of hardware saturation.
Common-bank Core/CUDA scores are comparable; different candidate mathematics
and different public families must not be relabeled equal-work benchmarks.

The host is a shared desktop. Pre/post snapshots retain CPU process activity,
GPU utilization/clocks/memory/power, allowed CPU set and AC state. They do not
prove absence of interference between snapshots. Until independent hardware
counters and an uncontaminated run support a stronger statement, retained
timings are **bounded shared-host diagnostics**, not release/performance gates.
No drivers, clock policy, affinity, power settings, or unrelated workloads are changed.

The bank is capped at 64 MiB of f32 values. Materialization also retains a
candidate-major temporary plus worker scratch; its conservative value-storage
bound is three bank payloads, excluding inputs, vector metadata and backend
resident copies. Full-bank materialization costs `O(rows * candidates)` storage
and upload, unlike the direct continuous centered-product descriptor route.
These costs are part of the feasibility answer, not hidden behind kernel timing.

The source accountant preserves the initial 600-line envelope result separately
and reports expansion, fixtures, wrappers, runner and documentation costs. It
does not reinterpret the original overrun or classify test infrastructure as
zero maintenance cost. Shipping source, public API, ABI, dependencies, versions
and distribution topology remain unchanged.

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

## Results

Local experiment results and exact source identity are recorded in the draft PR
after verification. No quality or saturation outcome is asserted before that run.
