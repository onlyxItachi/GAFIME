# Local RT lowering for canonical tabular regions

This is an unmerged, local-only follow-up to the
[tabular discovery milestone](v1.1-tabular-discovery.md), tracked by issue #75.
It is not a supported backend, a release artifact, a Gini implementation, or
evidence of a general RT speed advantage. The historical
[decision-path experiment](rt-gbdt-cuda.md) and
[paper reproduction record](rt-gbdt-paper-repro.md) remain separate evidence.

## Question and architectural boundary

Can the same accepted feature programs used by Core and ordinary CUDA lower
frozen region membership through OptiX without creating an RT semantic model?
Two useful workloads are ordinary decision-path-shaped conjunctions over source
columns and conjunctions over accepted algebraic atoms, such as softsign.
The latter must remain resident when reused; copying intermediate columns back
through Python or a fabricated legacy matrix handle is not an acceptable bridge.

Rust still owns mathematical identity, frozen thresholds, dependency validation,
evaluation context, evidence, selection and accepted-feature reuse. A local-only
executor option sends physical slots and the existing semantic program batch to
one optional entrypoint in `src/cuda/rt_abi.hpp`. The normal semantic ABI table
and public `gafime.semantic`/`EngineConfig` surfaces do not change.

Ordinary algebra and all evidence statistics retain the existing CUDA lowering.
Only eligible frozen-region materialization uses RT. The same Rust batch
marshaller supplies both routes, retaining descriptor lifetime and bank locking.
There is no RT-specific target/metric or candidate/evidence representation.

## Deliberately bounded first experiment

- Explicit local CMake opt-in and an actual RT-capable CUDA device are required.
- fp32 only. Mixed and fp64 fail before allocation, even though mixed stores f32.
- At most 65,536 rows, 256 regions per native call, 64 terms per region, and three
  distinct physical input axes per call. Rust greedily splits adjacent regions
  at batch/axis limits and before an input depends on an earlier output.
- A single wider region is unsupported, not silently rerouted to CUDA/Core.
- Referenced values and thresholds must be finite and non-subnormal; signed zero
  remains valid with exact `<=`/`>` semantics. This narrower execution envelope
  does not change the canonical predicate semantics.
- Overlapping regions are legal. All matching regions materialize membership;
  no first-hit/one-leaf assumption is applied to general feature candidates.
- The canonical registry rejects contradictory intervals before lowering. A
  direct physical-ABI caller can supply an empty interval: the shared geometry
  builder represents it as an impossible box, and the exact guard produces
  all-zero membership rather than reporting unsupported execution.
- Geometry, explicit scratch and intermediate membership are call-local. No
  geometry-cache or warm-residency acceleration claim is made.
- The caller supplies a temporary-buffer budget. Checked host/device planning
  covers explicit buffers including dense membership scratch; opaque driver and
  OptiX context overhead is not a process-memory guarantee.
- A failed call does not mark output slots initialized. Successful output stays
  in the same physical semantic bank for evidence and later accepted reuse.

The native lowering reuses the existing conservative ordered-float-bucket
custom-AABB preparation and exact intersection guard. Traversal is a culling
implementation, not the mathematical authority. The standard precision files
expose only an RT-free bank inspection/output-commit seam; RT device/launcher
logic stays in the dedicated RT files.

## Why not simply reuse the historical score bridge?

The legacy compact-score experiment combines membership with a resident target
and a small score vocabulary. That is useful historical research, but it is not
the canonical heterogeneous evidence model. Reinterpreting its target field as
an unlabeled objective would recreate the rejected bridge architecture.

This experiment instead materializes the canonical region column in a resident
bank. Existing Pearson/Spearman/NMI/graph primitives consume it exactly as they
consume ordinary CUDA output. This supports arbitrary combinations of the
currently contracted evidence channels, not arbitrary future objectives.

Dense output has a cost proportional to rows times regions. A historical
first-hit compact-score result that avoids this output is not a same-workload
comparator, and its all-pairs-equivalent denominator must not be presented as
the number of primitive tests performed by this experiment. General predicates,
tree induction, Gini scoring and promotion into normal packages remain separate
decisions under #75.

## Local reproduction

Use the active source checkout and a locally installed OptiX SDK. These commands
do not publish anything; normal package builds remain RT-disabled.

```bash
cmake -S src/cuda -B target/rt-batch-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DGAFIME_CUDA_BUILD_TESTS=ON \
  -DGAFIME_CUDA_RT_BUILD_MODE=on \
  -DGAFIME_OPTIX_INCLUDE_DIR="$OPTIX_INCLUDE" \
  '-DGAFIME_CUDA_ARCHITECTURES=89-real;89-virtual'
cmake --build target/rt-batch-native --parallel 2
ctest --test-dir target/rt-batch-native \
  -R gafime_cuda_rt_semantic_region_smoke --output-on-failure
GAFIME_CUDA_V1_LIB="$PWD/target/rt-batch-native/libgafime_cuda_v1.so" \
  cargo +1.97.1 test -p gafime-gpu-sys \
  --locked \
  --features local-cmake-experiment \
  --test local_cmake_experiment_numeric_domain \
  local_semantic_regions -- --nocapture
```

Use the actual architecture and matching compiler tuple for another host; an
`sm_89` build is not evidence for an untested GPU. The Rust fixture must load the
explicit current local payload; configured-but-incompatible payloads fail.
An unconfigured hardware-conditional test is not physical evidence.

After correctness, the ignored release-profile diagnostic compares identical
programs, output cardinality and membership bytes for Core's default Rayon
execution, ordinary CUDA and explicit RT. It uses bounded synthetic
1,024-row/32-region and 8,192-row/128-region cases, with source and accepted
softsign atoms. It reports three raw materialization/delivery samples per case.
Before each sample it explicitly clears cached materializations outside timing
and checks the completed RT-node count. Same-frame cache retrieval is therefore
not mislabeled as a repeated traversal or warm RT execution.
Materialization includes source upload and call-local geometry; delivery is
  reported separately. It does not isolate fixed-function traversal time.

```bash
GAFIME_CUDA_V1_LIB="$PWD/target/rt-batch-native/libgafime_cuda_v1.so" \
  cargo +1.97.1 test -p gafime-gpu-sys --release \
  --locked \
  --features local-cmake-experiment \
  --test local_cmake_experiment_numeric_domain \
  local_semantic_region_same_workload_diagnostic \
  -- --ignored --nocapture --test-threads=1
```

Do not override `RAYON_NUM_THREADS` in this production-Core comparison. Preserve
raw stdout, exact source/payload/executable hashes, compiler/runtime/device
identity, power state and competing-workload observations with any result.
Correctness can be checked on a shared desktop; contaminated timing cannot
support a comparative performance claim. No repeated campaign is required to
make a losing or inconclusive result look attractive.

## Evidence status and promotion boundary

The PR's exact-head review/validation record is authoritative for completed
checks. This source document specifies the experiment and reproduction, not an
evergreen assertion that later edits remain qualified. Compilation, physical
correctness and timing must be reported separately. No result from this draft
changes release packaging, semantic `auto`, the normal CUDA capability table,
or the status of #75.

## Next experiment: generic program reuse, not geometry reuse

The measured cold baseline remains available unchanged. A possible follow-up is
to retain only the generic OptiX context/module/pipeline and fixed SBT records,
while rebuilding every bank-derived box, point buffer, acceleration structure
and transient workspace. This is a proposal, not an implemented mode or a
profiler-proven speedup.

The existing native device-state lease and execution mutex can provide lifetime
and concurrency ownership. Its legacy custom-AABB program must not be reused
wholesale: that object also owns feature/target-dependent execution caches.
A dedicated generic-program member and the existing explicit device-state
release mechanism would keep those responsibilities separate. The Rust local
executor would need to retain the existing payload/device state owner, not just
the bank or loaded library.

Persistent SBT records remain explicit device bytes. A reused mode must report
and reserve their footprint across subsequent bank allocation and non-RT
operations, not merely charge them during the next RT call or relabel them
opaque driver overhead. That is the resource-accounting decision to settle
before implementation; no new semantic catalog, persistent geometry engine or
standard ABI change is justified by this experiment.

The next test should distinguish one-time generic initialization from per-call
GAS construction, use changed inputs in different banks, exercise simultaneous
same-device owners and final-owner teardown, reject under-budget requests
before allocation, and preserve exact Core/CUDA membership. Fresh execution
and cached materialization must remain separate measurement categories.
