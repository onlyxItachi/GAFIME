# Agent Skills

GAFIME's tracked skills provide repository-specific decision guidance and
bounded helper tools for users and contributors. They do not replace the
[repository contract](contract.md), expand a task's authorization, or create a
second source of product truth.

## Audience classes

Every `.claude/skills/*/SKILL.md` declares one `metadata.audience` value:

- `end-user`: using, configuring, or interpreting an installed GAFIME release;
- `contributor`: changing or reviewing the repository;
- `both`: diagnostics or environment workflows useful in either context.

Audience controls discovery and routing, not quality standards. Human,
agent-assisted, and autonomous contributions remain subject to the same
contracts, evidence, tests, review, provenance, and merge authority described
in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Resolve guidance from the right source

Contributor skills travel with the checkout. Use the skill, `AGENT.md` or
`CLAUDE.md`, contracts, tests, and specialist documentation from the same
branch and commit being changed. Do not combine an implementation from one ref
with contributor guidance fetched from another ref.

For end-user work in a source or editable development environment, use the
skills from that checkout. For a published installation:

1. identify the installed Core version with
   `importlib.metadata.version("gafime")`;
2. identify the matching canonical GitHub Release/tag under the repository's
   [version policy](releases/release-operations.md#version-identity); and
3. use the end-user skills from that tagged source.

This binds guidance to a release name rather than asking users to manage raw
commit SHAs. If the installed version and the available skill source cannot be
matched, disclose the mismatch and consult the installed version's public API
and release documentation; do not silently apply `main` guidance to an older
release. A skill with audience `both` follows the contributor rule in a
repository task and the end-user rule in an installed-product task.

Mutable publication, package, backend, and capability facts stay in their
authoritative sources. Skills should query the public API or route to
[release status](releases/STATUS.md), GitHub Releases, PyPI, the
[release artifact manifest](../.github/release-artifacts.json), or the relevant
contract instead of copying a dated snapshot.

## Contributor use

Select only the skill whose description matches the task. Contributor skills
state frozen boundaries, decision criteria, and the evidence needed to support
a claim while leaving file traversal, hypotheses, and reversible implementation
choices to the contributor. A fixed sequence belongs in a skill only when the
sequence is itself contractual.

The repository currently separates orientation, performance, numerical,
backend, and pull-request review guidance so that unrelated doctrine is not
loaded for every change. The active checkout's contract remains authoritative
if a skill and that contract disagree.

## Helper boundary

Skill scripts are optional development or user tools. They must remain bounded,
must not become a production runtime input, and must use supported public APIs
or validated repository interfaces. Running a helper does not prove physical
backend execution, numerical parity, release readiness, or a performance claim
unless the helper explicitly produces the evidence required by the governing
contract.
