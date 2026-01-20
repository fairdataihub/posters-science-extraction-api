# Installation Guide

Complete installation instructions for poster2json.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Standard Installation (Linux/macOS)](#standard-installation-linuxmacos)
- [Windows Installation](#windows-installation)
- [Installing pdfalto](#installing-pdfalto)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA CUDA-capable GPU with ≥16GB VRAM
  - ≥24GB recommended for running both models simultaneously
- **RAM**: ≥32GB system memory
- **Storage**: ~50GB for models and dependencies

### Software Requirements

- Python 3.10+
- CUDA 11.8+ with compatible NVIDIA drivers
- Git

## Standard Installation (Linux/macOS)

### 1. Clone the Repository

```bash
git clone https://github.com/fairdataihub/poster2json.git
cd poster2json
```

### 2. Create Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install pdfalto

See [Installing pdfalto](#installing-pdfalto) below.

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
```

## Windows Installation

Windows users have two options:

### Option A: Docker (Recommended)

Docker provides the simplest cross-platform experience. See [DOCKER.md](DOCKER.md) for complete instructions.

```bash
docker compose up --build
```

### Option B: WSL2

1. Install WSL2 with Ubuntu:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

2. Install NVIDIA CUDA support for WSL2:
   - Download from [NVIDIA CUDA WSL](https://developer.nvidia.com/cuda/wsl)

3. Follow the Linux installation steps above inside WSL2.

## Installing pdfalto

`pdfalto` is required for PDF text extraction with layout preservation.

### Option A: Build with Docker (All Platforms)

The easiest cross-platform method. Produces a Linux binary for Docker/WSL2 use.

```bash
# Clone pdfalto
git clone --recurse-submodules https://github.com/kermitt2/pdfalto.git
cd pdfalto

# Create build Dockerfile
cat > Dockerfile.build << 'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake git && rm -rf /var/lib/apt/lists/*
WORKDIR /pdfalto
COPY . .
RUN cmake . && make -j$(nproc)
EOF

# Build and extract binary
docker build -f Dockerfile.build -t pdfalto-builder .
container=$(docker create pdfalto-builder)
docker cp "${container}":/pdfalto/pdfalto ./pdfalto
docker rm "${container}"

# Move to poster2json
mv ./pdfalto /path/to/poster2json/executables/pdfalto
chmod +x /path/to/poster2json/executables/pdfalto
```

### Option B: Build from Source (Linux/macOS)

Requires `cmake` and a C++ compiler (gcc/clang).

```bash
git clone --recurse-submodules https://github.com/kermitt2/pdfalto.git
cd pdfalto
cmake .
make -j$(nproc)
# Binary at: ./pdfalto
```

### Option C: Pre-built Binary

Check [pdfalto releases](https://github.com/kermitt2/pdfalto/releases) for pre-built binaries.

### Configure pdfalto Path

The pipeline searches these locations automatically:

1. `PDFALTO_PATH` environment variable (recommended)
2. `./executables/pdfalto` (in repository)
3. System PATH (`which pdfalto`)
4. `/usr/local/bin/pdfalto`
5. `~/Downloads/pdfalto`

Set the environment variable:

```bash
export PDFALTO_PATH="/path/to/pdfalto"
```

Or add to your shell profile (`~/.bashrc`, `~/.zshrc`):

```bash
echo 'export PDFALTO_PATH="/path/to/pdfalto"' >> ~/.bashrc
source ~/.bashrc
```

## Verifying Installation

Run the test suite on the included example posters:

```bash
python poster_extraction.py \
    --annotation-dir ./example_posters \
    --output-dir ./test_output
```

Expected output:
- JSON files in `./test_output/`
- Console shows extraction progress and metrics

## Troubleshooting

### CUDA Not Available

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**
- Verify NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### pdfalto Not Found

```
WARNING: pdfalto not found, falling back to PyMuPDF
```

**Solutions:**
- Set `PDFALTO_PATH` environment variable
- Place binary in `./executables/pdfalto`
- Verify binary is executable: `chmod +x pdfalto`

### Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Close other GPU applications
- Use 8-bit quantization (automatic for <16GB GPUs)
- Process PDFs and images separately

### Model Download Issues

```
OSError: We couldn't connect to huggingface.co
```

**Solutions:**
- Check internet connection
- Set `HF_TOKEN` for gated models: `export HF_TOKEN="your_token"`
- Use offline mode with pre-downloaded models

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PDFALTO_PATH` | Path to pdfalto binary | Auto-detected |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use | All available |
| `HF_TOKEN` | HuggingFace API token | None |

## Next Steps

- [Docker Setup](DOCKER.md) - Container deployment
- [API Reference](API.md) - REST API usage
- [Architecture](ARCHITECTURE.md) - Technical details

