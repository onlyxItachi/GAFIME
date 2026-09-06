# Issue #73: native evidence feasibility experiment

Status: **branch-only design experiment; not a supported public API or a
release feature**. This does not complete [#73](https://github.com/onlyxItachi/GAFIME/issues/73)
or activate [#72](https://github.com/onlyxItachi/GAFIME/issues/72).

This document retains the first Core-only checkpoint. The separately authorized
[quality and CUDA-reuse extension](issue-73-cuda-quality-experiment.md) records
later scope, source cost and local evidence without rewriting that checkpoint.

The maintainer authorized this bounded experiment on 2026-09-06 in advance of
the full #73 specification. The authorization and its limits are recorded in
[the issue](https://github.com/onlyxItachi/GAFIME/issues/73#issuecomment-5559795172).
The original pause still governs the unselected remainder of that roadmap.
RC2 stabilization remains on its separate release/hardening lane. Nothing in
this document authorizes a merge, version transition, or publication.

## Question and decision rule

Can representative new candidate mathematics and unsupervised,
self-supervised, and hybrid evidence reuse the current native implementation
without excessive code growth, engine duplication, or ownership violations?
If not, which obstacle actually requires Candidate IR rather than an ordinary
new primitive or a smaller evidence protocol?

There are two different yes/no questions:

1. **Native extensibility:** can the bounded experiment fit the existing HPC
   building blocks? This is answered by executable code and measured source
   growth, not by a proposed architecture diagram.
2. **Full #72 prerequisite:** does failure of this experiment establish that
   all of #72 must precede #73? Missing kernels, a new buffer descriptor, or a
   user-facing API decision alone do not establish that conclusion.

The provisional cost envelope, chosen before implementing the probe, is:

- at most 600 nonblank, noncomment Rust implementation lines, counting demo
  orchestration conservatively but recording test lines separately;
- no new dependency, `unsafe`, duplicated engine, public Python export,
  production metric/family ID, native ABI change, or GPU payload edit;
- one candidate-parallel execution pass with reusable worker-local scratch
  and compact, deterministically ordered results;
- at least two new candidate transforms and three structurally different
  evidence channels, including a genuinely target-free call;
- explicit unsupported scope, numerical failures, and resource costs.

These numbers are a review aid, **not an accepted universal LOC limit**.
Readable validation and rationale must not be compressed to pass it. Report
both physical and code-line counts; excluded tests and documentation are still
maintenance cost. A Core-only result cannot establish all-backend integration
cost or production throughput.

## Measured outcome

**Native extensibility: yes for this bounded Core/mixed slice. Full Candidate
IR as a prerequisite: not demonstrated.** Two new pointwise forms support
three target-free evidence channels and an optional labeled channel, without
a fake target, an engine copy, or a production-file change. The existing native statistics and
Rayon scheduling remain usable. This is an implementation-feasibility result,
not a claim that the complete #73 product fits the current public protocol.

The conservative implementation count is **654 code lines / 706 physical
lines**, including the runnable demo. It **misses** the provisional 600-line
envelope by 54 lines; the envelope is not retrospectively increased. Separate
test, documentation, and accounting costs are reported by the reproduction
tool. Input/error/availability descriptions and validation account for much
of this code; the experiment does not exhibit whole-engine specialization or
cross-backend copying. The overrun is a scope-review point, not causal evidence
that a full compiler IR would remove those validation obligations.

The resulting recommendation is to specify a small evidence/input contract
before a production #73 slice. Do not mandate all of #72 based solely on these
LOC numbers. Selected-program identity, recursive reuse, and framework delivery
remain strong reasons to develop the corresponding #72 semantic layer; the
complete #73 taxonomy is not validated by this experiment. The maintainer can
still choose to coordinate #72/#73 in v1.1 for those product requirements.

Resource accounting is also bounded: for `n` rows, `d` columns, `e` supplied
edges and `m` labeled rows, input validation is `O(n*d + e + m)` expected time
(duplicate labels use a hash set). Per-candidate work is `O(n + e + m)`;
worker-local vectors use `O(n + m)` temporary storage and only compact rows are
collected. The fixed catalog has three candidates: this is not a search-space
or large-catalog scalability benchmark. Paired views are already materialized
inputs, so their construction/storage cost is **not** eliminated.

The correlation adapter performs a native centered-statistics preflight and
then calls the existing Pearson primitive, paying an additional reduction
pass to preserve explicit unavailable diagnostics. The new graph reduction is
scalar native Rust. No wall-clock speedup, allocation-free hot path, GPU
execution, or end-to-end learner benefit is claimed.

Twelve focused fixtures pass on both Rust 1.89.0 (MSRV) and 1.97.1. They cover
target-free use, paired views, analytic graph energy, constants, labeled-row
membership, malformed input, late pointwise overflow, graph-zero norm,
mixed-precision scalar-oracle parity, and one/four-worker result ordering.
The retained review and hosted-check states belong to the exact draft PR head,
not a self-updating source SHA in this document.

## Experiment, not a new product

The native probe lives under
[`crates/gafime-cpu/examples/issue73_probe/`](../crates/gafime-cpu/examples/issue73_probe/).
It links the existing Core crate but is not linked into the Python extension.
An integration-test entrypoint makes its checks run with ordinary Rust
workspace tests. No installed API accepts these experimental descriptors.

The concrete lane is **Core, mixed precision, finite f32 inputs**. Pointwise
transforms remain f32; statistical work and evidence rows are f64. This is not
a new precision profile and does not advertise fp32, fp64, CUDA, ROCm, or Metal
support for the experiment. Existing shipping profile support is unchanged.

Candidate examples are an identity control, absolute coordinate difference,
and softsign `x / (1 + abs(x))`. These are representative mathematical shapes,
not a claim that DL Method needs those exact transforms. They distinguish new
feature mathematics from merely applying another metric to an old candidate.
Their compact enum is a **small semantic descriptor**, not a claim that the
experiment contains no representation at all; it is not the compositional,
versioned, framework-deliverable Candidate IR proposed in #72.

The evidence stress cases are:

| Paradigm | Concrete measurement | Important boundary |
| --- | --- | --- |
| Unsupervised | Absolute correlation with an anchor column as a redundancy diagnostic | No task labels; not a measure of universal feature usefulness |
| Self-supervised | Correlation of the same candidate on aligned original/alternate views | Alignment and view creation are supplied by the fixture; no contrastive negatives or encoder training |
| Structure-oriented | Graph-neighbor variation normalized by degree-weighted signal energy | Uses supplied edges, not graph construction or a clustering-quality guarantee |
| Hybrid / partially labeled | Optional task correlation on explicitly selected labeled rows | Unlabeled entries must not become fake labels or enter this score |

Each result keeps evidence channels separate. No universal weighted utility,
statistical significance certificate, or downstream ranking promise is added.
Constant signals and unsupported or invalid inputs must not acquire an
apparently favorable score by silently substituting zero for an undefined
quantity. The exact formulas, edge policies, and tests in the probe are local
experimental definitions, not newly approved public numerical semantics.

For edges `(i, j, w)`, the graph channel is
`sum(w * (x_i - x_j)^2) / sum(w * (x_i^2 + x_j^2))`, with f32 signals and
weights widened for f64 statistical arithmetic. It is **not** a centered
Laplacian-score implementation: shifting every signal value changes its
denominator. Duplicate edges contribute their weights repeatedly; self-loops,
invalid endpoints, non-positive/non-finite weights, and an explicitly empty
graph are rejected. Constant candidates are reported unavailable to avoid a
trivial constant becoming a best-ranked feature. No production graph metric
or normalization policy is selected by this example.

## Reuse and the actual boundary

The existing Core slice-level
[`pearson_mixed`](../crates/gafime-cpu/src/kernels/precision.rs) and
[`pearson_sums`](../crates/gafime-cpu/src/simd/covariance.rs) primitives can be
called without constructing a resident matrix with a fabricated target.
This is an important distinction from the existing **production** execution
protocol. Correlation uses established runtime SIMD dispatch; new pointwise
and graph work is ordinary safe native Rust, not a claim of hand-vectorized
execution for every operation.

The current production boundary has real restrictions:

- [`GafimeLaunchProtocol`](../src/common/gafime_gpu_abi.hpp) carries feature
  combinations, metric IDs, chunks, ranking, and permutation descriptors;
  it does not carry an arbitrary feature program or evidence graph.
- Matrix upload uses a row-aligned target, validated by
  [`gpu_abi_impl.hpp`](../src/common/gpu_abi_impl.hpp). A zero/dummy target is
  not an implementation of target-free semantics.
- The current metric set is Pearson, Spearman, MI, and R2. Native backend
  dispatch is explicit, not a dynamically extensible evaluator interface.
- Existing time-series and decision-path expansion in
  [`family.rs`](../crates/gafime-orchestrator/src/family.rs) demonstrates native
  materialization reuse, not preservation of arbitrary feature-program
  semantics after materialization.
- CUDA/HIP execution graphs are command capture/replay, **not** data-neighbor
  graphs. Shared terminology must not become a false capability claim.

Consequently, a successful Core probe proves reusable *internals*, not that
`EngineConfig`, the resident lifecycle, exported reports, or the GPU ABI already
implement #73. Retrofitting evidence by silently reinterpreting an existing
metric ID or reserved field is not an acceptable shortcut.

## Full #73 scope ledger

The original issue deliberately did not freeze algorithms or public APIs.
This ledger accounts for its breadth without pretending that one prototype
implements every learning paradigm.

| Area | This experiment | Contract still needed before a product implementation |
| --- | --- | --- |
| Target-free evidence | Native redundancy and supplied-neighbor measurements | Evidence identity, normalization, missing-data policy, diagnostics and capability negotiation |
| Multi-view / self-supervision | Same candidate evaluated on two aligned views | View identity, permitted augmentations, alignment, sampling and train/evaluation boundaries |
| Hybrid / semi-supervision | Optional labeled-row channel alongside unlabeled channels | Label provenance, pseudo-label confidence, masks and consumer selection policy |
| Clustering, density, manifold, anomaly or graph structure | Supplied edges test a different data-access shape | Objective choice, neighborhood construction, weighting and scaling policy; no clustering algorithm selected |
| Reconstruction / masked prediction | Not implemented | Frozen fitted state, permitted inputs, leakage prevention and evaluation definition |
| Contrastive / cross-modal objectives | Not implemented | Positive/negative pairing, batch semantics, encoder ownership and native primitive needs |
| Temporal prediction | Not implemented by this probe | Causal windows, split boundaries, state reset and sequence alignment |
| Learned representations / auxiliary models | Not implemented | External learner versus GAFIME ownership, fitted-state identity and reproducible evaluation |
| Streaming / online updates | Not implemented | Epochs, bounded state, update/invalidation and result-comparability contracts |
| Heterogeneous evidence and selection | Separate compact channels only | Named/versioned schemas, policy validity, unavailable evidence and deterministic selection |
| Feature delivery and recursive reuse | Fixed in-memory candidate descriptions only | Stable semantics, dependencies, source/logical arity, frozen parameters and consumer execution; strongly aligned with #72 |
| GPU execution and profiles | Static boundary assessment only | Reviewed ABI/capability changes, native lowerings, precision-specific oracles and physical evidence |
| Scientific utility for DL Method | Not evaluated | Consumer-owned held-out experiments, leakage controls, comparable budgets and downstream metrics |

Unsupervised feature discovery need not mean unsupervised downstream learning.
Conversely, target-informed discovery does not authorize inference-time target
access. Any production feature-delivery contract must preserve that distinction.
The DL Method consumer motivates prioritization; this public experiment does
not disclose private research, freeze its method, or claim TabFM competitiveness.

## Sequencing after the experiment

If the bounded native slice fits, propose a reviewed, narrow #73 phase with an
explicit evidence and lifecycle contract. Do not require the full future
Python frontend/JIT or every #72 capability merely to expose a useful native
primitive. Equally, do not hide a second ad-hoc IR in a collection of adapters.

If preserving candidate meaning, heterogeneous evidence, consumer selection,
and delivery requires repeated schema/engine expansion, the maintainer's
conditional roadmap is: v1.1 establishes the necessary #72/#73 semantic spine;
v1.2 expands capabilities on that foundation. Record the concrete requirement
that triggers this choice. This document does not unilaterally change release
milestones or declare either issue complete.

The next product specification must settle: the first consumer acceptance
case, candidate/evidence schemas, optional-target semantics, ownership and
lifetimes, inference inputs, capability negotiation, numeric oracles, resource
budgets, and validation. Then update the issue's activation state for that
scope through the normal maintainer process.

## Reproduction and retained evidence

From this branch with the repository's release Rust toolchain:

```bash
cargo +1.97.1 test -p gafime-cpu --test issue73_evidence_feasibility -j 2
cargo +1.97.1 run -p gafime-cpu --example issue73_probe -j 2
python tests/release_measure/issue73_probe_scope.py --base <pre-experiment-commit>
```

The accounting tool reports physical lines and a transparent lexical code-line
measure (nonblank lines excluding line comments, retaining Rust attributes).
It checks that the branch changes only this experiment, its tests, accounting,
and documentation. It fails if production, ABI, dependency, or version files
change. It does not measure compiled size, throughput, or lines needed for a
future public API.

The probe's analytic and adversarial fixtures test implementation behavior;
they are not experiments demonstrating predictive gains. Thread-count parity
checks establish deterministic ordering/results, not a speedup. Source counts
and the final review/validation record are retained in the draft PR. The
candidate remains unmerged until the maintainer chooses the next scope.
