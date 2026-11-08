# CPython-Compatible Planner Randomness

The native combination planner uses the private
`crates/gafime-orchestrator/src/plan/cpython_mt19937.rs` module. This is the
existing CPython-compatible MT19937 implementation, renamed from `legacy_rng`
to describe an active v1 numerical/planning invariant rather than obsolete code.
It is not a new RNG or an optimization.

Seed words, initialization, rejection sampling, random-stream consumption and
shuffle order remain unchanged. Rust remains the sole owner of candidate
planning; Python does not supply shuffled candidate lists. The retained private
`PythonRandom` type is not a public API. No dependency, public seed parser,
candidate identity/order, profile, backend or ABI behavior changes.

Existing orchestrator goldens cover seeded feature order and descriptor order.
The rename adds an independent CPython oracle for a seed wider than u64 and two
successive shuffles on the same stream. Existing golden values and numerical
tolerances are not replaced. Focused orchestrator tests verify the same planner
contracts; installed-package family/seed checks remain part of normal PR CI.

This structural pruning does not qualify new release artifacts or make a
performance claim. Published RC3 source, packages and provenance remain unchanged.
