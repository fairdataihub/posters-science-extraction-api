# Docker Setup

This document describes how to run the poster extraction pipeline using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (≥8GB VRAM recommended)
- **Ollama running on the host** (the container communicates with host Ollama)

### Installing NVIDIA Docker Runtime

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

### Installing Ollama on Host

Ollama must be running on the host machine (not in Docker):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.1:8b-instruct-q8_0
ollama pull qwen3-vl:4b-instruct-q8_0

# Start Ollama server (if not running as service)
ollama serve
```

## Building the Image

```bash
docker-compose build
```

Or build directly:

```bash
docker build -t poster-extraction:latest .
```

## Running

### Production Mode

```bash
docker-compose up
```

This will:

- Process all posters in `./manual_poster_annotation/`
- Output results to `./output/`
- Connect to host Ollama for model inference

### Development Mode

For development with live code reloading:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This mounts the source code as a volume, so changes to `poster_extraction.py` are reflected immediately.

### Running a Single Command

To run the container interactively or with custom arguments:

```bash
docker-compose run --rm poster-extraction \
  python poster_extraction.py \
  --annotation-dir /app/input \
  --output-dir /app/output
```

## Server Deployment

### Prerequisites on Server

1. **Install NVIDIA Docker Runtime** (see Prerequisites section above)
2. **Install and configure Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3.1:8b-instruct-q8_0
   ollama pull qwen3-vl:4b-instruct-q8_0
   ```
3. **Verify GPU Access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
4. **Ensure sufficient resources**:
   - GPU: ≥8GB VRAM
   - RAM: ≥16GB recommended
   - Disk: ≥20GB free (for Ollama models)

### Deployment Steps

1. **Clone repository on server**:

   ```bash
   git clone <repository-url>
   cd poster-extraction-clean
   ```

2. **Build the image**:

   ```bash
   docker-compose build
   ```

3. **Configure volumes** (update paths in `docker-compose.yml` as needed):

   - Input directory: Where posters are stored
   - Output directory: Where results will be written

4. **Start the service**:

   ```bash
   docker-compose up -d
   ```

5. **Monitor logs**:

   ```bash
   docker-compose logs -f poster-extraction
   ```

## Configuration

### Environment Variables

You can override environment variables via `.env` file or directly in `docker-compose.yml`:

- `CUDA_VISIBLE_DEVICES`: Which GPU to use (default: `0`)
- `PDFALTO_PATH`: Path to pdfalto binary (optional, auto-detected if in PATH)
- `OLLAMA_HOST`: Ollama server URL (default: `http://host.docker.internal:11434`)

Create a `.env` file in the project root:

```bash
CUDA_VISIBLE_DEVICES=0
OLLAMA_HOST=http://host.docker.internal:11434
```

### Volumes

The docker-compose setup mounts:

- `./manual_poster_annotation` → `/app/input` (read-only)
- `./output` → `/app/output` (read-write)

**For server deployment**, update volume paths to absolute paths:

```yaml
volumes:
  - /data/posters/input:/app/input:ro
  - /data/posters/output:/app/output
```

### Connecting to Host Ollama

The container needs to reach the Ollama server running on the host. On Linux, add this to your docker-compose.yml:

```yaml
services:
  poster-extraction:
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
```

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Docker runtime is installed and verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Cannot Connect to Ollama

Ensure Ollama is running on the host and accessible:

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Verify models are available
ollama list
```

### pdfalto Build Fails

If pdfalto fails to build in the Dockerfile, you can:

1. Use a pre-built binary by setting `PDFALTO_PATH` environment variable
2. Comment out the pdfalto build section and install it manually
3. Use PyMuPDF fallback (the code will automatically fall back if pdfalto is unavailable)

### Out of Memory

If you encounter OOM errors:

- Ensure Ollama has sufficient GPU memory
- Reduce batch size or process fewer posters at once
- Adjust memory limits in `docker-compose.yml`

## Notes

- Ollama manages model loading and GPU memory
- The first run will be faster if models are already cached in Ollama
- pdfalto is built during Docker image creation for PDF processing
