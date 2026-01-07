# ---- builder: compile pdfalto ----
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System deps needed to build pdfalto
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake autotools-dev autoconf automake libtool \
    libpoppler-dev libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

# Faster / more reliable git
RUN git config --global http.postBuffer 524288000 \
    && git config --global http.lowSpeedLimit 0 \
    && git config --global http.lowSpeedTime 0 \
    && git config --global http.version HTTP/1.1

# Pin to a known version tag (change if you want another)
ARG PDFALTO_TAG=0.4

# Shallow clone + shallow submodules
RUN --mount=type=cache,target=/root/.cache/git \
    git clone --depth 1 --branch ${PDFALTO_TAG} \
    --recurse-submodules --shallow-submodules \
    https://github.com/kermitt2/pdfalto.git /tmp/pdfalto \
    && cmake -S /tmp/pdfalto -B /tmp/pdfalto/build \
    && cmake --build /tmp/pdfalto/build -j"$(nproc)" \
    && cp /tmp/pdfalto/build/pdfalto /usr/local/bin/pdfalto


# ---- runtime: your API image ----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Only runtime libs (no compiler toolchain)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libpoppler-dev libxerces-c-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Bring in pdfalto binary
COPY --from=builder /usr/local/bin/pdfalto /usr/local/bin/pdfalto

WORKDIR /app

# Install python deps with pip cache
COPY requirements-prod.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

COPY poster_extraction.py api.py ./

RUN mkdir -p /app/input /app/output

EXPOSE 8000
CMD ["python", "api.py"]
