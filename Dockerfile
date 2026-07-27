# ----------------------------
# Runtime / app
# ----------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Install runtime deps + python
# python3.10-dev provides Python.h, which Triton needs at runtime to JIT-compile
# its cuda_utils extension module on the GPU.
# Install runtime deps + python.
# python3.10-dev (Python.h) and build-essential (gcc/g++/make) are REQUIRED at
# RUNTIME, not just build: bitsandbytes routes the quantized model through
# Triton, which JIT-compiles a cuda_utils.c helper on first inference and links
# -l:libcuda.so.1. Without the Python headers and a compiler that gcc step fails
# with "returned non-zero exit status 1" and every upload errors out. Keeping the
# toolchain in the image makes that compile succeed regardless of which
# bitsandbytes/triton version is resolved.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
# IMPORTANT: rebuild with `docker build --no-cache` when poster2json has a new release
COPY requirements-prod.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir --upgrade poster2json

# Copy application code (poster extraction logic comes from poster2json library)
COPY config.py api.py job_worker.py validation.py poster_extraction_schema.json poster_schema.json ./

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Expose API port
EXPOSE 8000

# Set default command to run API server
CMD ["python", "api.py"]
