# Local RT weighted-region and frontier experiment

This is a bounded continuation of the [compact conditional experiment](rt-semantic-compact-experiment.md),
not RT promotion or a new supported Python API. These two adaptations are
implemented on draft PR #97. They establish correctness and lifecycle reuse,
not a comparative performance conclusion.

## Scope decision

Two adaptations reuse existing machinery closely enough for this pass:

- A weighted sum of frozen region memberships. Rust owns its canonical program
  identity and coefficients; Core is the reference. Local CUDA/OptiX reuses the
  existing query membership state and writes only the selected output.
- Exact two-/three-objective frontier selection for the explicit local fp32
  executor. Rust validates evidence, applies constraints, defines directions,
  handles ties and ranks accepted candidates. The native query only counts
  points inside physical regions.

Entropy/information-gain and regression accumulators are not included: their
logarithmic or floating reduction behavior needs separate numerical evidence.
General reusable multi-policy indexing and neighborhood/graph construction
also remain later experiments with distinct lifetime and algorithm contracts.
No existing feature, score, or selection behavior is weakened to fit RT.

## Weighted regional program

Conceptually, for declared regions R and frozen weights w,
`feature(row) = sum_r w[r] * membership(R[r], row)`.
The sum has a canonical region order rather than traversal-callback order;
overlap is legal. Weights are mathematical input, not selection scores or
learned parameters inferred by the backend. The all-one count feature remains
available under its established contract. Arithmetic, dependency, resource and
inference behavior are validated independently from any speed measurement.

The Rust declaration is `CandidateRegistry::region_weighted_sum(regions,
weights)` (also available through a discovery round); fp64 uses
`region_weighted_sum_f64`. Inputs are 2–64 distinct decision regions, further
bounded by the registry's logical/transitive arity and depth limits. Canonical
sorting preserves each region/weight pair. Finite negative and zero weights are
legal; duplicates and profile-mismatched weights are rejected. Weights are f32
for fp32/mixed materialization, f64 for fp64. Core is the reference in all three
profiles; only fp32 has this local RT lowering. Nonfinite output fails closed.

The optional native endpoint
`gafime_gpu_semantic_region_query_materialize_weighted_sum_rt_v1` consumes a
successfully executed query and a fresh output slot. Overlap uses retained
membership bits; the internally proven first-hit partition path retains a
region ordinal per row instead. Both sum from positive zero in submitted
canonical region order, never floating atomics in traversal order. The direct
path preserves the original coverage result by mapping nonzero ordinals to 1.
Additional device temporary storage is `4 * region_count + 4` bytes (copied
weights and a validation flag); query, output bank and host staging reservations
remain separately charged. Failed validation/overflow never commits the slot.

The executable cookbook is
[`region_aggregate_lifecycle`](../crates/gafime-gpu-sys/tests/local_cmake_experiment_numeric_domain.rs):
two overlapping rules with weights +2 and -3 produce an actual feature column;
the selected feature becomes a predicate atom in the next round and executes on
new unlabeled row identities. It is not an evidence alias.

## Frontier lowering

For a minimize objective use its original fp32 value as a coordinate. For a
maximize objective use the exact negation. A possible dominator of candidate
`i` lies in the intersection `coordinate[j] <= coordinate[i][j]` over all
objectives. One existing compact query returns each such region's occupancy.
Subtracting the multiplicity of the exactly equal objective vector distinguishes
weak dominance from the required strict improvement on at least one objective.
Signed zeros compare equal, as in the canonical Rust policy.

This is not a score approximation or rank-position-to-float encoding. It does
not narrow mixed/fp64 evidence to fp32. Unsupported profiles, shapes or numeric
domains fail closed on explicit local RT selection. The normal Core and GPU
selection routes keep their established behavior.

The query may still retain overlap-safe quadratic membership bits. This pass
does not claim subquadratic worst-case memory/time, and keeps checked work and
byte admission. Sorting-based conventional frontiers remain important
comparators; beating an exhaustive pairwise loop alone would not establish the
best selection algorithm.

The caller uses the existing `SelectionPolicy::pareto_objectives` and
`SemanticSession::accept_with` with the explicit
`GpuBackend::local_compact_rt_semantic_executor`. No second selection policy is
introduced. A narrow optional executor hook receives a `ParetoFrontierRequest`
containing profile, shape and physical coordinates only; Rust consumes returned
weak counts and removes equal-vector multiplicity before its established final
ranking. Ordinary executors keep the existing Core policy path without creating
this coordinate bank. The local envelope is at most 8,192 eligible candidates,
two or three objectives, finite non-subnormal fp32 coordinates, and the existing
quadratic work cap plus explicit memory admission. Constraints and missingness
are resolved before coordinates are formed. RT does not discover the policy,
learn objective weights, or replace all GAFIME selection operations.

An earlier evidence query is released before the call-local frontier query;
the evidence table and retained feature bank remain reserved. Frontier geometry
is not cached across policies in this pass. The test
`local_pareto_selection_matches_core_with_conflicts_ties_and_inference` compares
two-/three-channel hybrid workflows against Core through accepted reuse and
inference. A separate quadratic test oracle validates equal vectors, signed
zeros, direction changes and invalid native counts; it is not a production
fallback or a benchmark competitor.

## Validation boundary

Focused Core weighted tests cover canonical pairing, profile precision,
dependency limits, finite subnormals and overflow. Native physical smoke tests
cover RequireRT and conventional CUDA routes, overlapping and triangle-eligible
regions, multi-axis groups, invalid weights/counts/budgets, output-slot
initialization, retry after overflow, and caller-device restoration. Selection
unit tests cover exact ties/directions, constraints/missingness, byte/work limits,
unsupported profiles/dimensions and malformed weak counts. The inherited
session tests continue to guard foreign and stale evidence provenance.

Reproduce the local physical suite only with a configured CUDA/OptiX build:

```sh
ctest --test-dir target/rt-batch-native --output-on-failure
GAFIME_CUDA_V1_LIB="$PWD/target/rt-batch-native/libgafime_cuda_v1.so" \
  cargo +1.97.1 test -p gafime-gpu-sys --features local-cmake-experiment \
  --test local_cmake_experiment_numeric_domain --locked -- --test-threads=1
```

The PR's exact-head review/evidence record binds final local results and hashes.
Hosted checks are reported separately; a local pass is not hosted approval.
No timed speedup is claimed: concurrent host activity and correctness fixtures
are not performance evidence. In particular, an RT frontier can do more work
than a conventional sorted sweep; that comparison remains unmeasured here.
No RC2, main, package identity, standard ABI or publication change is authorized.
