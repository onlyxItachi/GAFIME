# Contributing to GAFIME

GAFIME is a multi-language native project. Python provides the public API,
Rust owns the PyO3 boundary, planning, CPU kernels, orchestration, reporting,
and backend selection. CUDA, Metal, and ROCm/HIP own their native GPU kernels,
launchers, runtime calls, and graph/replay implementation details.

This guide is for source development. Release tagging and publication are
maintainer-controlled and should not be started without explicit approval.

## Local Development

Use a project-local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
python -m pip install -e ".[dev,sklearn]"
```

With `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,sklearn]" --no-build-isolation
```

The root build produces Core only. GPU payload development builds are staged
and installed explicitly after Core; there are no root-build skip or strictness
switches for CUDA or ROCm.

CUDA payload:

```bash
python .github/scripts/stage_gpu_payload.py cuda payload-src/gafime-cuda
uv pip install --python .venv/bin/python -e payload-src/gafime-cuda \
  --no-build-isolation
```

ROCm/HIP payload with an explicit supported offload target:

```bash
python .github/scripts/stage_gpu_payload.py rocm payload-src/gafime-rocm \
  --rocm-wheel-policy system
GAFIME_ROCM_ARCHS=<rocm-offload-target> \
  uv pip install --python .venv/bin/python -e payload-src/gafime-rocm \
    --no-build-isolation
```

## Verification

Run focused checks before committing native changes:

```bash
python -m gafime --check
python -m pytest tests/python/test_v1_public_truthfulness.py -q
python -m pytest tests/python/test_v1_rocm.py -q
```

ROCm and CUDA tests that require device access may skip in restricted
environments. Do not count sandbox-only device skips as hardware validation.

## Developer Docker Images

Docker is for maintainers and contributors who need reproducible source-build
environments. Docker images in this repository are not distribution images.
Normal users should install the published artifacts documented in `README.md`;
the prebuilt thin ROCm wheel comes from the matching GitHub Release rather than
PyPI.

CUDA development environment:

```bash
docker compose run --build gafime-cuda-dev
```

CPU/Core smoke environment:

```bash
docker compose run --build gafime-core-smoke
```

The CUDA developer image includes:

- CUDA Toolkit 13.3 base image,
- compiler toolchain,
- Rust,
- CMake/Ninja,
- Python development headers,
- GAFIME `dev`, `sklearn`, and `bench` Python dependencies.
- locally staged `gafime-cuda` payload installation by default.

Extra workstation packages can be layered without editing the Dockerfile:

```bash
docker build \
  --build-arg EXTRA_PIP_PACKAGES="torch" \
  -t gafime:cuda-dev .
```

The Core smoke image skips CUDA and ROCm and verifies the Rust/PyO3 CPU path.

## Source Layout

- `python/gafime/`: Python public API and thin v1 adapter.
- `crates/`: Rust ownership for Python boundary, planning, CPU kernels, reporting, and GPU C ABI loading.
- `src/cuda/`: CUDA kernels.
- `src/metal/`: Apple Metal backend.
- `src/rocm/`: ROCm/HIP kernels.
- `.claude/skills/`: optional maintainer/agent skills, classified and bound to
  checkout or release truth as documented in
  [`docs/agent-skills.md`](docs/agent-skills.md).
- `docs/`: release notes, backend notes, validation logs, and reference docs.

## Pull Requests And AI Review

`main` remains protected and accepts tracked changes only through a pull request. The required GitHub approving-review count is zero; independent human approval is not required. `@onlyxItachi` is the sole final merge authority.

Before merge, every PR must have a current-head AI Review Record submitted as a GitHub review, all configured required status checks reported for the final head after executing against GitHub's current PR merge commit for that head/base pair, and all review conversations resolved. A `COMMENTED` review is valid review evidence; an `APPROVED` review state is not required. The AI Review Record must state the model, role, exact reviewed commit SHA, verdict, and findings.

Keep changes reviewable and semantically focused. Independent architecture,
feature, and fix work should use separate focused or explicitly stacked PRs;
mutually dependent documentation, validation, and implementation needed to
make one change coherent may remain together. Validation is proportionate to
the affected surfaces: execution changes need the relevant backend-local and
end-to-end evidence, while documentation/governance-only changes do not require
unrelated benchmark campaigns.

Autonomous and AI-assisted contributions follow the same contracts, tests,
evidence, review, provenance, safety, numerical, and release gates as every
other contribution. They do not receive weaker gates or acquire an extra
human-authorship or approving-review requirement.

Contributor skills may help an agent orient, change a backend or numerical
path, assess performance evidence, or review a PR. They are doctrine and
evidence guidance from the current checkout, not a substitute for engineering
judgment or permission. Human, co-developed, and fully autonomous work uses the
same merge standard; using a skill neither grants authority nor creates a new
gate. See [`docs/agent-skills.md`](docs/agent-skills.md).

Use this record shape:

```markdown
### AI Review Record
- Model: <model name>
- Role: <review role>
- Reviewed commit SHA: <40-character head SHA>
- Verdict: PASS | CHANGES_REQUIRED
- Findings: None | <findings and dispositions>
```

The reviewed SHA must equal the current PR head. A later head commit invalidates the record and requires a new review. A base change invalidates the merge-commit CI evidence and requires the configured checks to run against the new merge commit. A merge-blocking verdict or unresolved blocking finding prevents merge.

Intermediate PR commits do not need to be green. Merge eligibility is based on the final reviewed head and required checks that execute against GitHub's current PR merge commit for that exact head/base pair. Workflows configured for `main` must then validate the resulting commit on `main`; a failure blocks release use and follow-on integration until it is corrected or reverted through another PR.

## Release Branches

Candidate stabilization uses protected branches named
`release/v<canonical-semver>`. A branch is cut only from a green `main` commit;
creating it is not a version bump, tag, build publication, or release.

Release-branch work is limited to the named candidate's bounded stabilization
and uses focused pull requests, normal merge commits, exact-head AI review,
strict checks, resolved conversations, and the same maintainer authority as
`main`. Durable fixes normally land on `main` first and are backported with
provenance (prefer `git cherry-pick -x` on a temporary branch). An urgent
release-first fix must be present on `main` no later than final admission; the
admission merge normally supplies that forward-port without a duplicate PR. Do
not merge divergent `main` wholesale into a release branch.

Once the candidate is settled, its exact release-branch tip is the build,
freeze, tag, and publication source. Admit that unchanged tip to `main` before
tagging by creating a temporary admission branch from current `main`, merging
the release tip into it, and submitting that branch through an ordinary strict
PR. Do not merge `main` into the release branch. Publication additionally
requires the frozen build, tag, and current release tip to resolve to the same
SHA. After accepting the frozen candidate and before admission or tagging,
install exact-ref update/deletion protection; retain that read-only lock after
publication. A later source fix requires a deliberate unlock, reviewed PR, new
exact-tip build, and restored lock. The lifecycle and permitted change matrix
are in
[`docs/releases/release-branches.md`](docs/releases/release-branches.md).

Canonical `v*` release tags are covered by layered rulesets: an authorized
creation-only rule and a separate no-bypass update/deletion rule. After
admission, create the tag on the exact accepted tip, verify that both rules
apply, and dispatch publication from the tag ref. The publisher pins downstream
checkouts to the preflight source SHA and revalidates live tag/branch identity
before every irreversible upload lane.

## Release Safety

Do not create or push release tags, dispatch publication, or publish to PyPI
without maintainer approval. Follow
[`docs/releases/release-operations.md`](docs/releases/release-operations.md).

`.github/workflows/build_wheels.yml` validates and freezes an immutable bundle;
it never publishes. After that exact source is reviewed, frozen, and admitted
into `main`, a canonical tag may bind the same release-branch commit. The
manual-only `.github/workflows/publish_release.yml` verifies and publishes the
byte-identical frozen bundle in Core-first order, runs public exact-version
installation checks, and creates the GitHub Release last. Pushing a tag alone
neither builds nor publishes a release.
