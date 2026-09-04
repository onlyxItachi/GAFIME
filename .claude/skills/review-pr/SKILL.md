---
name: review-pr
description: Review a GAFIME pull request against its exact current head, merge-commit checks, repository contracts, and evidence claims, then produce the required AI Review Record.
metadata:
  audience: contributor
---

# Review a GAFIME Pull Request

Review the actual current diff and the active checkout's contracts; do not adopt
the pull-request narrative as the conclusion. Confirm the live head SHA, base,
mergeability, review threads, and configured checks. Evidence bound to an older
head is historical unless the relevant bytes and source identity are proven
unchanged.

Determine the affected public API, ABI, numerical, ownership, safety, backend,
package, documentation, and release surfaces. Verify claims using the evidence
class that can support them: tests for behavior, independent oracles for
numerics, physical execution for hardware runtime claims, and frozen provenance
for release artifacts. A green narrow test does not excuse an uncovered
contract boundary.

Classify findings clearly:

- a blocking finding identifies a correctness, safety, compatibility,
  architecture, integrity, or required-evidence defect that must be resolved;
- a non-blocking suggestion is an improvement that does not invalidate the
  change or its claims.

Do not manufacture findings to make a review appear substantive, and do not
downgrade a demonstrated defect because remediation is inconvenient. Confirm
that review conversations are resolved and that required checks executed
against GitHub's current merge commit for the exact head/base pair.

Produce the AI Review Record specified in `CONTRIBUTING.md`, including model,
role, exact reviewed commit SHA, verdict, and findings/dispositions. A later
head commit invalidates the record. A base change invalidates the merge-commit
CI evidence and requires the configured checks to run for the new pair.

Review is read-only unless the task separately authorizes submitting the record,
resolving a thread, editing code, or merging. The absence of a required human
approval does not transfer final merge authority away from the maintainer.
