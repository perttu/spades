# SPAdes with GPU Support Dockerfile
# Multi-stage build for optimized production image

# Build stage with CUDA development tools
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Fix repository keys and install build dependencies
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --allow-unauthenticated \
    ca-certificates \
    gnupg \
    lsb-release && \
    apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    libbz2-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    autoconf \
    automake \
    libtool \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    setuptools \
    wheel

# Create working directory
WORKDIR /opt/spades

# Copy source code
COPY . .

# Create build directory and configure with GPU support
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DSPADES_GPU_SUPPORT=ON \
          -DSPADES_CUDA_SUPPORT=ON \
          -DSPADES_GPU_DEBUG=OFF \
          -DCMAKE_INSTALL_PREFIX=/opt/spades/install \
          -DCMAKE_CXX_STANDARD=17 \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          ..

# Build SPAdes with GPU support
RUN cd build && \
    make -j$(nproc) && \
    make install

# Verify CUDA installation and GPU kernels
RUN cd build && \
    nvidia-smi || echo "No GPU available for testing" && \
    nvcc --version

# Production stage with CUDA runtime only
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/spades/install/bin:${PATH}
ENV PYTHONPATH=/opt/spades/install/share/spades

# Fix repository keys and install runtime dependencies
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --allow-unauthenticated \
    ca-certificates \
    gnupg \
    lsb-release && \
    apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libbz2-1.0 \
    zlib1g \
    libncurses5 \
    libncursesw5 \
    liblzma5 \
    libssl3 \
    libffi8 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib

# Copy built SPAdes from builder stage
COPY --from=builder /opt/spades/install /opt/spades/install

# Create directories for data
RUN mkdir -p /data/input /data/output /data/tmp
WORKDIR /data

# Set permissions
RUN chmod +x /opt/spades/install/bin/*

# Create non-root user for security
RUN useradd -m -u 1000 spades && \
    chown -R spades:spades /data /opt/spades
USER spades

# GPU environment setup
ENV SPADES_GPU_SUPPORT=1
ENV SPADES_GPU_STRATEGY=sorted
ENV SPADES_GPU_BATCH_SIZE=1000000
ENV SPADES_GPU_MEMORY_FRACTION=0.8

# Health check to verify installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.path.append('/opt/spades/install/share/spades'); import spades" || exit 1

# Expose potential ports for monitoring (optional)
EXPOSE 8080

# Default command
CMD ["bash", "-c", "echo 'SPAdes with GPU support ready!' && echo 'GPU Status:' && nvidia-smi || echo 'No GPU detected' && echo 'Usage: docker run --gpus all -v /path/to/data:/data spades-gpu spades.py --help' && /bin/bash"]

# Labels for metadata
LABEL maintainer="SPAdes Team"
LABEL description="SPAdes genome assembler with GPU acceleration support"
LABEL version="4.0.0-gpu"
LABEL cuda.version="12.3"
LABEL gpu.support="enabled" 