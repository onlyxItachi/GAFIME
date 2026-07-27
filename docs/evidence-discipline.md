# Evidence And Claims Discipline

The contract already says what must be true before a change lands: bit parity or an
approved tolerance, a measurable benefit, no undocumented numerical difference. This
document defines **how that evidence is produced and how it is stated**, so that two
reviewers reach the same verdict from the same artifact.

It applies to every PR that changes a kernel, a numerical path, a scheduling decision,
or any release-facing claim. It is normative: `docs/contract.md`, `CLAUDE.md`, and
`AGENT.md` carry the binding summary under **Evidence And Claims Discipline**, and
`tests/release_measure/contract_00_policy_files.py` gates its presence.

Nothing here replaces the release-measure gates. Those validate the installed package
end to end across backends. This document governs the narrower question a kernel or
perf PR must answer: *is this measurement real, and does the claim match it?*

---

## Part 1 - Verification before claiming

### V1. Parity gates timing, structurally

A benchmark harness must verify bit-exact agreement with the reference **before** it
reports any timing, and must refuse to report timings for a variant that failed parity.

This is an ordering requirement, not a suggestion. A faster wrong answer is worse than
a slow right one, so verification cannot be the step that happens after the number is
already attractive. Put the parity result at the top of the output.

### V2. Inputs derive from the reference's branch structure, not from a distribution

Every conditional in the reference implementation is a mandatory input class. Read the
scalar/reference helper, enumerate its branches, and construct one input per branch.
Random or uniform data is an *additional* input, never the primary one.

For floating-point kernels the minimum corpus is: `NaN`, `+Inf`, `-Inf`, `-0.0`, `0.0`,
negative values, subnormals (`MIN_POSITIVE`, `MIN_POSITIVE / 2`), values at and adjacent
to each clamp boundary, and extreme finite magnitudes. For any chunked or vectorized
kernel, also sweep lengths that exercise the tail: exactly one full chunk, one chunk
plus one element, and an unaligned length.

Uniform random data is the weakest possible parity input, because the values that break
a kernel are exactly the values a distribution almost never produces.

### V3. Baselines are copied verbatim, never paraphrased

When comparing against shipped code, copy the shipped function byte for byte into the
harness and record its source path, line, and commit in a comment. If the baseline is
reimplemented from memory or from its documentation, the measurement describes the
reimplementation.

### V4. One mechanism per variant

If a change bundles N mechanisms, the harness must contain N+1 variants so each
mechanism's contribution is separable. Reporting only the combined figure produces a
correct number attached to a wrong conclusion, and the wrong conclusion is what gets
generalized into the next change.

A bundled result may not be attributed to one of its mechanisms.

### V5. Caller trace before impact

Before a measurement is described as an improvement, establish where the measured code
runs: which caller, how often, and whether it is on a per-candidate, per-permutation,
or one-shot path. Record it in the PR body in one line.

A measured speedup on code with no production caller is not an improvement. This check
is cheap and it is the one most often skipped.

### V6. Measurement hygiene

Required for any reported figure:

- pin the process to a core; record load average before the run
- report best-of-N, and say so; if tail behaviour matters, report p50 and p99 instead
- prevent elimination and constant folding of both inputs and outputs
- build with the release profile actually shipped (`lto`, `codegen-units`, `opt-level`)
- sweep sizes that cross cache levels, from L1-resident to well past last-level cache
- state the host: CPU/GPU model, thread count, ISA rung exercised, toolchain version

The size sweep is not optional. A kernel's bottleneck changes with working-set size, and
a single size routinely misstates a result by a large factor in either direction.

### V7. A negative claim requires a positive control

Before using a search, a grep, or an absence of output to prove that something is *not*
present, confirm that the same command finds something known to be present. A search
that silently matched nothing — a bad pathspec, the wrong ref, a mistyped pattern —
produces empty output that is indistinguishable from a true absence.

State the control next to the negative claim so a reader can see the search worked:

```console
$ git grep -c 'string_known_to_be_present' <ref> -- <same pathspec>
13
$ git grep -n 'string_believed_absent' <ref> -- <same pathspec>
(no output)
```

This rule exists because a reviewer asserted that this repository contained no handling
for a required release operation, having run a search whose pathspec matched nothing
against the given ref. The handling was present, documented in the runbook, and gated by
a release contract. The claim was retracted in place; the cost was a recommendation to
build something that already existed, and a maintainer's time spent reading it.

A negative claim is often the most load-bearing thing in a review, because it is what
authorizes new work. It deserves the strongest evidence, not the weakest.

---

## Part 2 - Claims and truthfulness

### C1. A measured claim carries its conditions

A performance or accuracy claim is incomplete without: host and microarchitecture,
thread count, the ISA rung or backend actually exercised, input distribution, the
statistic reported, and the sizes measured. A number without conditions is not a claim
that can be checked, reproduced, or falsified.

### C2. Name what is not claimed

State the boundary explicitly: which architectures, backends, ISA rungs, thread counts,
input distributions, and API levels were **not** measured. Kernel-level results must say
so rather than implying an end-to-end figure.

### C3. Agreement is not correctness

When two implementations share a computation stage, agreement between them is not
evidence that either is right. A defect in the shared stage appears as perfect
agreement.

This is a real, recurring failure in this project. CPU and GPU both build interaction
products in `fp32`. When that product overflows, both produce the same wrong value, and
a cross-backend differential test reports agreement. A reviewer once concluded a
covariance path was "clean up to 1e19" on exactly this basis; the measurement had shown
agreement, not correctness. Establish correctness against an independent oracle of
higher precision or different construction - never against a peer that shares the
suspect stage.

### C4. Volunteer the regression

If a change improves one path and degrades another, state the degradation in the same
artifact, with the same specificity as the improvement. A reviewer who discovers an
unmentioned trade-off must reasonably discount everything else in the report.

### C5. Retract in place

A claim later found to be wrong is corrected where it was made - the PR review, the
issue, the doc - not silently dropped. Record what the earlier measurement actually
showed and why it did not support the claim. Corrections are first-class evidence of a
working process, not embarrassments to minimize.

### C6. Release notes carry non-claims and evidence boundaries

Every release note must include:

- **Deliberate Non-Claims** - capabilities a reader could reasonably infer that the
  release does not provide, including known mathematical limits that remain unfixed.
- **Evidence Boundaries** - which platforms, backends, and hardware the release was
  actually validated on, and which validations are pending or provisional.

A provisional or unapproved numerical tolerance must be named as such in the release
note of any release that ships the affected backend to users.

---

## Part 3 - Review checklists

### PR author

- [ ] parity verified before timing, and reported first
- [ ] hostile-input corpus covers every branch of the reference (V2)
- [ ] baseline copied verbatim, with source and commit recorded (V3)
- [ ] one variant per mechanism; no bundled attribution (V4)
- [ ] caller trace stated in one line (V5)
- [ ] host, threads, ISA rung, statistic, sizes stated (C1)
- [ ] what was not measured is stated (C2)
- [ ] any regression the change introduces is stated (C4)

### Reviewer

- [ ] could this result be explained by a shared stage rather than correctness? (C3)
- [ ] is the measured code actually reachable from a production caller? (V5)
- [ ] is a single-size or single-host result being generalized? (V6, C1)
- [ ] does any claim in the docs, README, or release note exceed the evidence?
- [ ] is an invariant being documented where it could instead be made unrepresentable?
- [ ] does this change operate on the same layer as the problem it cites?

The layer question catches a specific and expensive error: a change that achieves a
stated goal by acting on the wrong layer. A packaging change offered as the remedy for a
documentation or numerical-disclosure problem should be suspicious on its face, as
should a documentation change offered for a correctness problem. Check the goal *and*
the layer — a reviewer who checks only the goal will approve a fix that trades one
problem for a larger one, and will do it while reporting the goal as met.

The invariant question is a design preference with teeth: prefer an interface where a
precondition cannot be violated over one where it is merely written down. Passing a
pre-sliced buffer beats documenting a length requirement on a count parameter, because
the documented version decays the moment a second caller appears.

### Release

- [ ] Deliberate Non-Claims section present and current (C6)
- [ ] Evidence Boundaries section present and current (C6)
- [ ] every provisional tolerance either approved with recorded evidence, or the
      affected backend declared experimental
- [ ] version strings agree across manifests and at runtime
- [ ] every claim about a prior release is verifiable against the tag and the index

---

## Part 4 - Known gaps in this document's own coverage

Stated here so the document is subject to its own rules:

- The shared hostile-input corpus exists for the CPU fixed-bin binning path only.
  Promoting it to a fixture used by CPU and GPU parity tests alike is not done.
- No perf ratchet exists. The Regression Policy forbids reducing correctness but nothing
  measures throughput regression between commits.
- Clippy is not gated in CI, so lint quality is unenforced.
- Documented safety invariants on `unsafe` blocks are not lint-enforced.
- The SSE4.2 and NEON ISA rungs ship but have no recorded kernel measurements, despite
  ARM wheels being distributed.

---

## Appendix - Worked example: why V2 is written the way it is

A candidate optimization replaced a branchy scalar bin conversion with a branchless
vector sequence. The reference is:

```rust
fn fixed_bin_from_scaled(scaled: f32, max_bin: u32) -> u32 {
    if scaled.is_nan() || scaled <= 0.0 { 0 }
    else if !scaled.is_finite() || scaled >= max_bin as f32 { max_bin }
    else { scaled as u32 }
}
```

The natural vector form is truncate-then-clamp: `cvttps_epi32`, `max` against zero,
`min` against `max_bin`. It is three instructions, it is obviously correct, and it is
wrong.

`cvttps_epi32` maps `NaN`, `-Inf`, **and `+Inf`** all to `INT_MIN`. Clamping therefore
sends `+Inf` to bin `0`, where the reference sends it to `max_bin`. Recovering the
distinction requires an explicit ordered compare, which also returns false for `NaN` and
so leaves that case correct:

```rust
let ge = _mm256_cmp_ps(scaled, max_bin_f, _CMP_GE_OQ);
let t  = _mm256_blendv_epi8(t, max_bin_i, _mm256_castps_si256(ge));
```

Uniform random input never contains `+Inf`, so the three-instruction version passes any
distribution-based parity test at any sample count. Only a corpus derived from the
reference's own branches - which forces `+Inf` to appear - separates the two. That is
the entire reason V2 requires branch-derived inputs rather than a distribution.
