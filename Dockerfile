# =============================================================================
# GAFIME CUDA Source-Build Development Image
#
# This image is for maintainers who want a reproducible Linux/NVIDIA source
# build environment. It intentionally uses the same setup.py native build path
# as local development instead of hand-compiling individual legacy artifacts.
#
# Usage:
#   docker build -t gafime:cuda-dev .
#   docker run --gpus all --rm gafime:cuda-dev
#
# Requirements:
#   - NVIDIA Container Toolkit for runtime GPU access
#   - Docker BuildKit recommended
# =============================================================================

ARG CUDA_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG EXTRA_PIP_PACKAGES=""

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/gafime-venv \
    PATH="/opt/gafime-venv/bin:/root/.cargo/bin:${PATH}" \
    GAFIME_SKIP_ROCM=1 \
    STRICT_CPU=1 \
    STRICT_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    curl \
    git \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /workspace/GAFIME
COPY . .

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel pybind11 cmake \
    && python -m pip install --no-build-isolation -e ".[dev,sklearn,bench]" \
    && if [ -n "$EXTRA_PIP_PACKAGES" ]; then python -m pip install $EXTRA_PIP_PACKAGES; fi

HEALTHCHECK --interval=60s --timeout=20s --retries=3 \
    CMD python -m gafime --check || exit 1

CMD ["python", "-m", "gafime", "--check"]
