FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    autotools-dev \
    autoconf \
    automake \
    libtool \
    libpoppler-dev \
    libxerces-c-dev \
    python3.10 \
    python3-pip \
    curl

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Build pdfalto
RUN git clone https://github.com/kermitt2/pdfalto.git /tmp/pdfalto && \
    cd /tmp/pdfalto && \
    git submodule update --init --recursive && \
    cmake . && \
    make && \
    chmod +x pdfalto && \
    cp pdfalto /usr/local/bin/pdfalto

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-prod.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy application code
COPY poster_extraction.py api.py ./

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Expose API port
EXPOSE 8000

# Set default command to run API server
CMD ["python", "api.py"]

