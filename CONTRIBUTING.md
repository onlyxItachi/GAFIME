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

Useful build controls:

- `GAFIME_SKIP_CUDA=1`: skip CUDA build.
- `GAFIME_SKIP_ROCM=1`: skip ROCm/HIP build.
- `STRICT_CPU=1`: fail if the Rust/PyO3 CPU runtime cannot build.
- `STRICT_CUDA=1`: fail if CUDA cannot build.
- `STRICT_ROCM=1`: fail if ROCm/HIP cannot build.
- `GAFIME_ROCM_ARCHS=<rocm-offload-target>[,<rocm-offload-target>...]`:
  explicit HIP targets.

Examples:

```bash
GAFIME_SKIP_CUDA=1 GAFIME_SKIP_ROCM=1 STRICT_CPU=1 \
  uv pip install --python .venv/bin/python -e . --no-build-isolation

GAFIME_SKIP_CUDA=1 GAFIME_ROCM_ARCHS=<rocm-offload-target> STRICT_CPU=1 \
  uv pip install --python .venv/bin/python -e . --no-build-isolation
```

## Verification

Run focused checks before committing native changes:

```bash
python -m gafime --check
python -m pytest tests/test_v045_native_spine.py -q
python -m pytest tests/test_rocm_native_backend.py -q
```

ROCm and CUDA tests that require device access may skip in restricted
environments. Do not count sandbox-only device skips as hardware validation.

## Developer Docker Images

Docker is for maintainers and contributors who need reproducible source-build
environments. Docker images in this repository are not distribution images.
Normal users should install GAFIME from PyPI wheels.

CUDA development environment:

```bash
docker compose run --build gafime-cuda-dev
```

CPU/Core smoke environment:

```bash
docker compose run --build gafime-core-smoke
```

The CUDA developer image includes:

- CUDA Toolkit 13.2 base image,
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
- `.claude/skills/`: maintainer/agent helper skills.
- `docs/`: release notes, backend notes, validation logs, and reference docs.

## Pull Requests And AI Review

`main` remains protected and accepts tracked changes only through a pull request. The required GitHub approving-review count is zero; independent human approval is not required. `@onlyxItachi` is the sole final merge authority.

Before merge, every PR must have a current-head AI Review Record submitted as a GitHub review, all configured required status checks reported for the final head after executing against GitHub's current PR merge commit for that head/base pair, and all review conversations resolved. A `COMMENTED` review is valid review evidence; an `APPROVED` review state is not required. The AI Review Record must state the model, role, exact reviewed commit SHA, verdict, and findings.

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

## Release Safety

Do not create tags, push release tags, enable release workflows, or publish to
PyPI without maintainer approval.

The wheel workflow may be intentionally disabled during backend development.
Check `.github/workflows/build_wheels.yml` before assuming tag pushes will
publish artifacts.
