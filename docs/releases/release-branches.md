# Candidate Release Branches

Candidate release branches are protected, bounded stabilization lanes. Their
settled tips are the exact candidate and release sources; `main` may continue
to accept unrelated reviewed work while a candidate stabilizes.

## Naming And Creation

Candidate branches use:

```text
release/v<canonical-semver>
```

The first branch under this policy is `release/v1.0.0-rc.2`. Canonical version
syntax remains governed by the repository release-version tooling and
[release operations](release-operations.md).

Cut a candidate branch only from a green `main`, after its required checks have
completed. Record the fork point in the release work. Do not cut one from a
feature branch or an uncommitted local tree. Protect it against deletion,
force-pushes, and unreviewed direct changes while it is active.

Creating the branch does **not**:

- change Cargo, Python, or documentation version identity;
- create or authorize a tag, PyPI publication, or GitHub Release;
- establish a settled, release-eligible frozen candidate; or
- prove that the initial branch tip is ready to release.

The creation push may run validation and produce an immutable validation
bundle. That bundle is not publication authority: only the later settled,
version-correct, reviewed, and admitted release tip can supply the eligible
frozen release bundle.

Version identity changes require their own focused, reviewed preparation.

## Stabilization Scope

Only bounded candidate stabilization belongs on a release branch: release
identity and notes, focused correctness or security fixes, regression tests,
and packaging or release-validation corrections. Feature development, broad
refactors, new architecture, and unrelated cleanup remain on `main` and must
not enter merely because a release branch exists.

Every tracked change uses a focused pull request into the candidate branch and
a merge commit. It requires the current-head AI Review Record, strict required
checks against the current PR state, and no unresolved blocking review thread.
Candidate urgency does not weaken ordinary numerical, ABI, security,
packaging, or release contracts.

## Fix Flow And Divergence

Prefer durable fixes on `main` first, then selectively integrate the exact
reviewed fix into the candidate branch. If an urgent candidate-only condition
requires a release-first fix, that fix must be present on `main` no later than
final admission. The admission merge normally supplies that forward-port, so a
duplicate pre-admission PR is unnecessary. Release branches must not accumulate
orphaned fixes.

Never merge a divergent `main` wholesale into the release branch. Selectively
integrate only the reviewed fixes required for the candidate so unrelated
development cannot silently expand its scope or change its release source.

## Settle And Freeze The Candidate

After stabilization and exact-head review are complete, record the protected
release-branch tip as the candidate source. Run final qualification and Build
and Validate Wheels against that exact tip. An eligible frozen build must be a
`push` or `workflow_dispatch` run whose `head_branch` is exactly
`release/$tag`, where `$tag` is the planned canonical tag (for example,
`release/v1.0.0-rc.2`). Pull-request synthetic merge builds are not release
sources.

Do not advance the release branch after freezing. Any required tracked change
creates a new tip and invalidates the previous bundle. Once the frozen
candidate and its release-blocking evidence are accepted, install an exact-ref
ruleset that blocks updates and deletion with no bypass before final admission,
tagging, or publication. If a source fix is later required, deliberately remove
that exact lock, land the reviewed release-branch PR under the normal candidate
rules, build and verify the new exact tip, then restore the lock. Never treat an
unlocked or changed tip as the previously accepted candidate.

## Admit The Exact Tip To Main

Admission establishes that the exact frozen release tip is part of current
repository history without pulling an independently advancing `main` into the
release branch:

1. Cut a temporary admission branch from the current green `main`.
2. Merge the exact settled release tip into that temporary branch with a merge
   commit. Do not change the release branch.
3. Open the admission branch as a pull request to `main` so strict checks and a
   current-head AI Review Record evaluate the real current integration.
4. Merge the admission pull request with a merge commit after all required
   checks pass and blocking threads are resolved.
5. Verify that the unchanged release tip is now an ancestor of `main`.

If conflict resolution or a failing gate reveals a release-source defect, fix
and review it on the release branch, freeze the new exact tip, and rebuild the
temporary admission integration. Do not hide a release fix only in the
admission branch.

The final `main` commit is an integration result, not the artifact source. The
unchanged settled release tip remains the authoritative build, tag, and
publication source.

## Tag And Publish

Only after admission may the canonical tag be created on the exact frozen
release tip and the frozen bytes published. Before publication, mechanically
verify all of these identities:

- the build run event is `push` or `workflow_dispatch`;
- the build run `head_branch` equals `release/$tag`;
- the remote release-branch tip equals the tag commit and authoritative build
  source SHA; and
- that SHA is an ancestor of current `main`.

```text
green main -> release/v<canonical-semver> -> settle -> build/freeze exact tip
current main -> temporary admission branch + exact release tip -> PR -> main
exact release tip (now ancestor of main) -> tag same tip -> publish frozen bytes
```

## Retention

Retain the pre-publication exact-ref lock after publication so the release
branch remains a read-only provenance record. Do not reuse it for another
version, resume development on it, move its history, or delete it as routine
cleanup.

The admission merge normally forward-ports every release-first fix by making
the release tip an ancestor of `main`. If an exceptional release-only fix is
made later, forward-port it through a reviewed `main` pull request; never
rewrite a published tag or unlock and move the retained release branch.

See also the [current release-train status](STATUS.md), the
[artifact matrix](release-artifact-matrix.md), and
[contribution governance](../../CONTRIBUTING.md).
