# GAFIME v0.5.0 Multi-Feature Discrete Region Architecture

## Summary

Build v0.5.0 around native first/second-order discrete search plus
beam-grown higher-order regions. Keep `k=1` and `k=2` as true native searches,
because second-order is still GPU-manageable and must not depend on univariate
winners. For `k>=3`, avoid full enumeration and grow from strong lower-order
regions one gate at a time.

This is a planning artifact only. Do not implement this plan until the user
explicitly resumes v0.5.0 implementation work.

## Public API And Defaults

Add developer/debug region objects while keeping normal user flow through
`EngineConfig` and `ComputeBudget`:

- Add `DiscreteGate(feature_index, kind, threshold=None, interval=None, direction="ge", scale=1.0, sharpness=12.0)`.
- Add `DiscreteRegionCandidate(gates, value_feature=None, mode="soft", source_order, source="native"|"grown", parent_id=None, candidate_id="")`.
- Keep v0.4.0 `DiscreteFunctionCandidate` and family helpers working; internally convert old threshold/interval/rectangle candidates into the region representation where useful.
- Keep `EngineConfig.metric_names` as report metric control; no new discrete metric override.
- Keep `discrete_ranking="split_aware"` as the default selector path.

Add `ComputeBudget` controls:

```python
max_native_discrete_order = 2
max_grown_discrete_order = 4
max_native_discrete_feature_pairs = 25_000
max_pair_regions_per_feature_pair = 32
discrete_region_beam_width = 512
max_region_extensions_per_parent = 128
max_discrete_candidates_per_order = 100_000
max_discrete_results = 10_000
max_validated_discrete_candidates = 2_000
```

Keep existing v0.4 fields for compatibility. Treat
`max_feature_pairs_for_rectangles` as the legacy rectangle cap; the new region
planner uses `max_native_discrete_feature_pairs`.

## Implementation Architecture

- Split planning into two stages:
  - native stage: generate all first-order gates and budgeted all-pair second-order regions directly from native feature pairs;
  - growth stage: for orders `3..max_grown_discrete_order`, extend top previous-order regions with one additional gate using beam search.
- Build a gate library from existing quantile thresholds and intervals only. No tree-derived thresholds and no learnable thresholds in v0.5.0.
- Native second-order candidates must include pair regions like threshold-threshold masks, threshold-interval masks, interval-interval masks, and optional `value_feature * pair_mask` variants.
- Higher-order candidates use `child_mask = parent_region_mask * soft_gate(new_feature)`, but initially recompute masks from gate descriptors on GPU instead of caching parent masks.
- Ranking flow:
  - use CUDA/CPU split-aware selector scores for candidate ordering;
  - update residual baseline after each order using a lightweight NumPy ridge baseline over continuous results plus retained discrete winners;
  - score final kept candidates with user-selected `metric_names`;
  - run stability/permutation only on top retained discrete candidates, capped by `max_validated_discrete_candidates`.
- Reporting:
  - region results should use `family="discrete_region"` for new multi-gate candidates;
  - include `source_order`, `source`, `selection_score`, and gate descriptors in `params`.

## GPU, Metal, Rust, And CPU Backends

- Add generic variable-arity soft-region native APIs:
  - `gafime_discrete_region_soft_batch_cuda`
  - `gafime_discrete_region_selection_batch_cuda`
- Pack candidates with offset arrays:

```text
candidate_gate_offsets[n_candidates + 1]
gate_feature_indices[n_gates]
gate_kind_codes[n_gates]
gate_directions[n_gates]
gate_params[n_gates, 2]
gate_scales[n_gates]
candidate_value_features[n_candidates]
candidate_sharpness[n_candidates]
```

- CUDA evaluates each candidate as a short gate loop. Branching on gate kind is uniform per candidate block, so it should not create warp-divergent tree traversal.
- Keep GPU hard mode rejected with the existing error. CPU/NumPy may evaluate hard regions with exact boolean masks.
- Add Metal soft-region parity with the same descriptor/offset design. If Metal region kernels are not ready in the first v0.5 implementation pass, Metal must fall back cleanly to CPU/NumPy for region candidates and document the gap.
- Extend Rust cache-local scheduling to variable feature sets with a template id based on `(region_order, gate_kind_signature, value_flag)`. Reuse current `BatchScheduler.order_equations` behavior where possible.

## Test Plan

- Unit tests:
  - gate and region candidate serialization, `combo`, params, expression text;
  - CPU soft/hard region evaluation for orders 1-4;
  - old v0.4 `DiscreteFunctionCandidate` compatibility.
- GPU parity:
  - CUDA soft-region stats vs Python reference for order 1, 2, 3, 4;
  - CUDA selector scores vs Python reference;
  - GPU hard mode still raises the exact existing error.
- Planner tests:
  - second-order native planner finds pair-only/XOR-style signals where first-order gates are weak;
  - third/fourth-order planner uses beam growth and never full cubic/quartic enumeration;
  - candidate caps and result caps produce warnings, not silent truncation.
- Application benchmarks:
  - synthetic 3-way AND/region regression and classification;
  - pair-only interaction dataset proving native second-order search is not top-1D dependent;
  - California Housing residual-region benchmark against v0.4.0 and tree baselines;
  - Friedman1 sanity check.
- Profiling:
  - NCU profile for the generic region selector kernel;
  - compare recompute-vs-v0.4 rectangle kernel throughput, L2 hit rate, DRAM throughput, register count, spills, branch efficiency.

## Persistence And Release Notes

- `codex.md` should record that v0.4.0 is released to GitHub and PyPI.
- v0.5.0 target is multi-feature discrete regions.
- User-approved decisions:
  - public debug API,
  - budgeted all-pair native order 2,
  - recompute-first higher-order growth.
- No GAFIME DL method scope.
- GPU remains soft-only.
- v0.4.x remains patch-only: packaging fixes, docs corrections, and small benchmark/report polish.
