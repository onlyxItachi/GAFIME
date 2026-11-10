# Stable Source Integration And Previous Attempt Retirement

This records source preparation, not final stable qualification, artifact
freeze, admission or publication. The [mutable status](STATUS.md) owns current
gates; [release branches](release-branches.md) and
[release operations](release-operations.md) remain binding.

## Previous Unpublished Attempt

The earlier `release/v1.0.0` diverged from the RC3-qualified mainline. It was
not a published canonical stable release, and its versioned source must not be
merged wholesale into the current runtime. At retirement it had 25 commits not
in post-pruning `main`; `main` had 48 not in that attempt.

Under the maintainer's explicit retirement authorization, GitHub's supported
branch-rename operation preserved the complete history:

| Identity | Preserved value |
| --- | --- |
| Original branch | `release/v1.0.0` |
| Archive branch | [`archive/v1.0.0-previous-stable-attempt`](https://github.com/onlyxItachi/GAFIME/tree/archive/v1.0.0-previous-stable-attempt) |
| Original and archived commit | `70a08f2ef712591dd28951ff7c4d3db6bfcb6566` |
| Original and archived tree | `c5c82c5f00ac32a8b78e1b9024ea93dd9b765656` |
| Preservation ruleset | `26899443`, exact archive ref; no-bypass update/deletion/force-push restrictions |

The archive protection was installed before rename. All six pre-existing
release/main/tag rulesets remained unchanged; no rule was temporarily disabled
and no bypass or force-push was used. The local clean historical worktree was
renamed and retargeted to the archive without changing files or commits. A
verified complete-history Git bundle additionally preserves the original ref.
The old canonical ref was verified absent using Git's remote ref listing and
the raw Git-ref API, not branch-URL redirects.

At retirement, canonical `v1.0.0` Git tag and GitHub Release were absent, all
three PyPI `1.0.0` version endpoints were absent, and no open PR depended on
the old branch. These are operation-time observations, not reusable publication
preflight. Refresh every mutable prerequisite before any later release.

Published RC3 remains unchanged at
`b510b4dd9af4c47602846b9a58702ed005fe67e4`, with its retained branch, tag and
artifact provenance. No other historical branch is retired by this operation.

## Reconciled Preparation

The new preparation starts from green post-pruning `main`
`f49935ac635d8987781a1f8e4109ee1e5b98fa74`, including PRs #120–#123.
The earlier dirty `work/v1.0.0-stable-integration` worktree and its four reviewed
patches remain preserved rather than reset, copied wholesale or silently mixed.

- Reuse #108's intent for stable identity, metadata, current docs and exact
  version/tag tests, adapting it to current source and the pruned API.
- #109/#110's cap-warning fix and RC3 ingestion, numerical/lifecycle and GPU
  hardening are already present; do not reapply their old code.
- Replace the old unresolved branch-migration proposal with the completed
  archival record above; one focused source-preparation PR targets `main`.
- Preserve current pruning documentation, DataQualityAnalyzer's construction
  warning, unchanged OTSEncoder and the maintained public integrations.
- Keep historical release/evidence records, workflows, manifest, benchmark
  harness and production runtime unchanged. A Rust version unit-test
  expectation changes; no parser or runtime algorithm is replaced.

Stable versions are repository/Cargo `1.0.0`, Python/PyPI `1.0.0` and future
canonical tag `v1.0.0`. Payload metadata derives that same version and exact
`gafime==1.0.0` dependency through existing staging; no alternate authority is
added. Production/Stable classification identifies the target line, not a
completed final stable release.

## Approval And Candidate Handoff

The source-preparation PR requires current-head independent AI review, strict
current head/base merge-commit checks and resolved blocking threads. The
maintainer retains final approval and merge authority for this new PR.
Earlier pruning merge approval does not authorize its merge.

Only after that approval, merge and green `main` may the now-free canonical
`release/v1.0.0` be created from the exact green stable-prepared main commit.
Apply normal PR-only reviewed-change protection, strict required CI and
force-push/deletion restrictions, then independently review its settled tip.
Do not apply the final accepted-artifact lock prematurely.

Being ready to assess a source for Deep Security Scan does not mean it has
passed that scan. Exact-source security, final correctness/installed E2E and
performance evidence, DL Method, a new frozen stable bundle and physical
backend qualification remain separately authorized gates. So do final
locking/admission, canonical tagging and publication. No RC4 is introduced.
