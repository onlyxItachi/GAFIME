# =============================================================================
# GAFIME Docker Image - GPU-Accelerated Feature Interaction Mining Engine
#
# Multi-stage build:
#   Stage 1 (builder):  Compile CUDA kernels, C++ core, Rust backend
#   Stage 2 (runtime):  Slim image with compiled artifacts + Python
#
# Usage:
#   docker build -t gafime .
#   docker run --gpus all gafime python -c "import gafime; print('OK')"
#
# Requirements:
#   - NVIDIA Container Toolkit (for GPU access)
#   - Docker with BuildKit enabled
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — full toolchain for compiling all backends
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-dev \
    python-is-python3 \
    g++ \
    cmake \
    make \
    curl \
    git \
    ca-certificates \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Rust toolchain (for gafime_cpu PyO3 module)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /build

# Copy source code
COPY . .

# -------------------------------------------------------
# Build 1: CUDA kernels → libgafime_cuda.so
# -------------------------------------------------------
RUN echo "=== Building CUDA Backend ===" && \
    nvcc \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90,code=sm_90 \
    -gencode=arch=compute_90,code=compute_90 \
    -O3 --shared \
    -Xcompiler "-fPIC,-O3" \
    -I src/common \
    -o libgafime_cuda.so \
    src/cuda/kernels.cu && \
    echo "✅ CUDA backend built"

# -------------------------------------------------------
# Build 2: CPU backend → libgafime_cpu.so
# -------------------------------------------------------
RUN echo "=== Building CPU Backend ===" && \
    g++ \
    -O3 -fopenmp -shared -fPIC \
    -I src/common \
    -o libgafime_cpu.so \
    src/cpu/cpu_backend.cpp && \
    echo "✅ CPU backend built"

# -------------------------------------------------------
# Build 3: gafime_core (pybind11 C++ module)
# -------------------------------------------------------
RUN echo "=== Building gafime_core ===" && \
    pip3 install --no-cache-dir "pybind11[global]" && \
    cd gafime_core && \
    mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGAFIME_CORE_ENABLE_OPENMP=ON \
    -DGAFIME_CORE_USE_FETCHCONTENT=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3.11 \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && \
    make -j$(nproc) && \
    cp gafime_core*.so /build/ && \
    echo "✅ gafime_core built"

# -------------------------------------------------------
# Build 4: gafime_cpu (Rust PyO3 module)
# -------------------------------------------------------
RUN echo "=== Building gafime_cpu (Rust) ===" && \
    pip3 install --no-cache-dir maturin && \
    cd src/cpu/gafime_cpu && \
    maturin build --release --interpreter python3.11 && \
    pip3 install --no-cache-dir target/wheels/gafime_cpu*.whl && \
    echo "✅ gafime_cpu built"

# -------------------------------------------------------
# Install Python package
# -------------------------------------------------------
RUN pip3 install --no-cache-dir numpy>=1.24 polars>=0.20 && \
    pip3 install --no-cache-dir -e .


# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with only what's needed to run
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime + OpenMP (no dev headers needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy compiled native libraries from builder
COPY --from=builder /build/libgafime_cuda.so /app/
COPY --from=builder /build/libgafime_cpu.so /app/

# Copy gafime_core compiled module
COPY --from=builder /build/gafime_core*.so /app/

# Copy Python package
COPY --from=builder /build/gafime /app/gafime
COPY --from=builder /build/setup.py /app/
COPY --from=builder /build/pyproject.toml /app/
COPY --from=builder /build/requirements.txt /app/
COPY --from=builder /build/README.md /app/
COPY --from=builder /build/LICENSE /app/

# Copy examples
COPY --from=builder /build/examples /app/examples

# Install Python deps + package
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
RUN pip3 install --no-cache-dir numpy>=1.24 polars>=0.20 && \
    pip3 install --no-cache-dir -e .

# Copy Rust module from builder's site-packages
COPY --from=builder /usr/local/lib/python3.11/dist-packages/gafime_cpu /usr/local/lib/python3.11/dist-packages/gafime_cpu

# Set library search path for native backends
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check — verify GAFIME loads with GPU
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python3 -c "from gafime.backends.fused_kernel import FusedKernelWrapper; print('OK')" || exit 1

# Default command
CMD ["python3", "-c", "import gafime; print('GAFIME ready'); from gafime.backends import resolve_backend; print('Backends OK')"]
