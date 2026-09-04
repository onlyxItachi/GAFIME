---
name: repository-orientation
description: Orient a GAFIME repository contribution by locating its authoritative contracts, ownership boundaries, and required evidence without freezing replaceable internals.
metadata:
  audience: contributor
---

# Orient a Repository Contribution

Use the active checkout as one coherent source of contributor truth. Begin with
its `AGENT.md` or `CLAUDE.md`, then read only the task-relevant sections of
`docs/contract.md`, specialist documentation, tests, and release gates. GitHub
state is authoritative for live branches, pull requests, issues, and checks;
historical evidence describes only the source state to which it was bound.

The task defines scope and authorization. Repository guidance constrains how
that work may be implemented; this skill grants no additional permission to
change code, external state, issues, releases, or infrastructure.

## Classify the affected boundary

Before defending or replacing an existing design, determine which category the
checkout actually assigns it:

- Public API, frozen ABI, numerical semantics, distribution identity, release
  integrity, and explicit ownership or safety rules are contracts. Preserve
  them unless the task explicitly authorizes a reviewed contract change.
- Internal representations, algorithms, and file-local abstractions are not
  frozen merely because they exist. They may change when no public or normative
  promise depends on them and evidence supports a better design.
- Historical documents and old implementations are evidence, not automatic
  current policy. Follow them only when a current normative source points to
  them.

Do not preserve a poor internal seam by creating a parallel ownership model,
and do not call a surface internal until exports, consumers, compatibility
tests, package contents, and contracts support that conclusion.

## Justify the change

A material architectural deviation should make its reasoning reviewable:

- identify the affected contract and consumers;
- state what remains invariant and what intentionally changes;
- explain why the existing internal shape is insufficient;
- provide correctness, compatibility, and affected-runtime evidence
  proportionate to the change; and
- isolate independent work into focused or explicitly stacked pull requests.

Use the domain-specific contributor skill when the change affects performance,
numerics, or a native backend. If satisfying the task requires crossing an
ownership, ABI, numerical, safety, or release boundary that was not authorized,
stop that path and surface the decision to the maintainer.
