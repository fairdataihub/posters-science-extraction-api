# Use NVIDIA CUDA base image for GPU support
# Using CUDA 11.8 with cuDNN for optimal compatibility
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    libpoppler-dev \
    libxerces-c-dev \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Install pdfalto dependencies and build pdfalto
RUN git clone https://github.com/kermitt2/pdfalto.git /tmp/pdfalto && \
    cd /tmp/pdfalto && \
    git submodule update --init --recursive && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cp pdfalto /usr/local/bin/pdfalto && \
    chmod +x /usr/local/bin/pdfalto && \
    rm -rf /tmp/pdfalto

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY poster_extraction.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set default command
CMD ["python", "poster_extraction.py", "--annotation-dir", "/app/input", "--output-dir", "/app/output"]

