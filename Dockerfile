# ============================================
# Stage 1: Build stage - compile pdfalto
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    autotools-dev \
    autoconf \
    automake \
    libtool \
    libpoppler-dev \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

# Build pdfalto
RUN git clone https://github.com/kermitt2/pdfalto.git /tmp/pdfalto && \
    cd /tmp/pdfalto && \
    git submodule update --init --recursive && \
    cmake . && \
    make && \
    chmod +x pdfalto

# ============================================
# Stage 2: Runtime stage - minimal production image
# ============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only runtime dependencies
# Note: pdfalto may need runtime libraries - adjust based on actual requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy pdfalto binary from builder stage
COPY --from=builder /tmp/pdfalto/pdfalto /usr/local/bin/pdfalto

# Set working directory
WORKDIR /app

# Copy production requirements first for better caching
COPY requirements-prod.txt requirements.txt

# Install Python dependencies (production only)
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY poster_extraction.py .
COPY api.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Expose API port
EXPOSE 8000

# Set default command to run API server
CMD ["python", "api.py"]

