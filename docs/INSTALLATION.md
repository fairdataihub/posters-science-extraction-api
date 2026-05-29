# Installation Guide

Complete installation instructions for poster2json.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Standard Installation (Linux/macOS)](#standard-installation-linuxmacos)
- [Windows Installation](#windows-installation)
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

### Option A: pip install from GitHub (Recommended)

```bash
pip install git+https://github.com/fairdataihub/posters-science-extraction-api.git
```

This installs the API service and all dependencies, including the
[poster2json](https://github.com/fairdataihub/poster2json) library that provides
the extraction pipeline. Start the service with:
```bash
python api.py
```

### Option B: Clone and Install (Development)

```bash
git clone https://github.com/fairdataihub/posters-science-extraction-api.git
cd posters-science-extraction-api
pip install -e .  # Editable install
```

### Option C: Requirements Only

```bash
git clone https://github.com/fairdataihub/posters-science-extraction-api.git
cd posters-science-extraction-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation

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

## PDF Text Extraction

No separate binary is required. PDF text extraction is handled by the
[poster2json](https://github.com/fairdataihub/poster2json) library, which uses
`pdfplumber` (pure Python, installed automatically with the other dependencies)
and a recursive XY-cut reading-order reconstruction. When a page yields too
little text, it falls back to PyMuPDF. Image posters are handled by the Qwen2-VL
vision model.

## Verifying Installation

Verify the extraction library imports cleanly, then start the service:

```bash
python -c "from poster2json import extract_poster; print('poster2json OK')"
python api.py   # then, in another shell: curl http://localhost:8000/health
```

Expected output:
- `poster2json OK` printed with no import errors
- The API server starts and `/health` returns a success response

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
- Use offline mode with pre-downloaded models

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use | All available |

## Next Steps

- [Docker Setup](DOCKER.md) - Container deployment
- [API Reference](API.md) - REST API usage
- [Architecture](ARCHITECTURE.md) - Technical details

