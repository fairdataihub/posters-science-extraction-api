# ----------------------------
# Runtime / app
# ----------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Install runtime deps + python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3.10 \
    python3-pip \
    libpoppler-dev \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Copy pdfalto binary from executables folder
COPY executables/pdfalto /usr/local/bin/pdfalto
RUN chmod +x /usr/local/bin/pdfalto

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-prod.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py poster_extraction.py api.py job_worker.py validation.py poster_extraction_schema.json poster_schema.json ./

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Expose API port
EXPOSE 8000

# Set default command to run API server
CMD ["python", "api.py"]
