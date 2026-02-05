# Docker Setup

Docker deployment guide for poster2json, including Windows and WSL2 support.

## Table of Contents

- [Quick Start](#quick-start)
- [Windows Setup](#windows-setup)
- [Development Mode](#development-mode)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Build and Run

```bash
# Build the image
docker compose build

# Run with GPU support
docker compose up

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Test the API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/extract \
  -F "file=@poster.pdf"
```

## Windows Setup

### Prerequisites

1. **Windows 10/11 with WSL2**
   ```powershell
   wsl --install
   ```

2. **Docker Desktop for Windows**
   - Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Enable WSL2 backend in Docker Desktop settings
   - Enable GPU support (Settings → Resources → WSL Integration)

3. **NVIDIA GPU Support**
   - Install [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Latest NVIDIA drivers (≥470.x)

### Building pdfalto for Windows/Docker

Since pdfalto doesn't have native Windows binaries, build it via Docker:

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

# Build and extract
docker build -f Dockerfile.build -t pdfalto-builder .
container=$(docker create pdfalto-builder)
docker cp "${container}":/pdfalto/pdfalto ./pdfalto
docker rm "${container}"

# Move to posters-science-extraction-api executables folder
mkdir -p ../posters-science-extraction-api/executables
mv ./pdfalto ../posters-science-extraction-api/executables/
```

### Running on Windows

```powershell
# From PowerShell or Windows Terminal
cd posters-science-extraction-api
docker compose up --build
```

## Development Mode

For active development with hot reload:

```bash
# First time (builds image, downloads models ~16GB each)
docker compose -f docker-compose.dev.yml up --build

# After editing Python files, restart to apply changes
docker compose -f docker-compose.dev.yml restart

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Stop
docker compose -f docker-compose.dev.yml down
```

### What Dev Mode Provides

- **Volume mounts**: `poster_extraction.py` and `api.py` are mounted as volumes
- **Model caching**: Models are cached in a named volume (survives rebuilds)
- **Fast iteration**: Edit code locally, restart container to apply

### Clearing Model Cache

If you change the model IDs, clear the cache:

```bash
docker compose -f docker-compose.dev.yml down -v
docker compose -f docker-compose.dev.yml up --build
```

The `-v` flag removes named volumes, forcing fresh model download.

## Production Deployment

### Using Production Config

```bash
# Copy .env.example to .env and set values, then:
docker compose -f docker-compose-prod.yml up -d
```

### Environment Variables

Create a `.env` file (see `.env.example`). Required for the job worker:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `BUNNY_PRIVATE_STORAGE` | Bunny CDN storage URL (for job input files) |
| `BUNNY_PRIVATE_STORAGE_KEY` | Bunny storage AccessKey |

Optional:

| Variable | Description |
|----------|-------------|
| `APP_PORT` | Host port (default `47362`) |
| `BUNNY_PUBLIC_STORAGE`, `BUNNY_PUBLIC_STORAGE_KEY` | Public Bunny storage |
| `CUDA_VISIBLE_DEVICES` | GPU device(s), e.g. `0` or `0,1` |
| `POLL_INTERVAL_SECONDS` | Job poll interval (default `30`) |
| `RESTART_POLICY` | Container restart policy (default `unless-stopped`) |

### GitHub Actions (Deploy to GPU server)

The workflow `.github/workflows/deploy-main.yml` deploys on push to `main`. Configure these **GitHub repo secrets**:

**Required:**

- `SSH_HOST` – Deploy server hostname
- `SSH_USER` – SSH user
- `SSH_PRIVATE_KEY` – SSH private key (e.g. ed25519)
- `DATABASE_URL` – PostgreSQL connection string
- `BUNNY_PRIVATE_STORAGE` – Bunny storage URL
- `BUNNY_PRIVATE_STORAGE_KEY` – Bunny AccessKey

**Optional (have defaults or can be empty):**

- `APP_PORT` (default `47362`), `BUNNY_PUBLIC_STORAGE`, `BUNNY_PUBLIC_STORAGE_KEY`, `CUDA_VISIBLE_DEVICES`, `POLL_INTERVAL_SECONDS`

The workflow creates a `.env` on the server from these secrets and runs `docker compose -f docker-compose-prod.yml up -d --build`.

### Health Checks

The container includes health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' poster-extraction

# View health check logs
docker inspect --format='{{json .State.Health}}' poster-extraction | jq
```

## Configuration

### docker-compose.yml

```yaml
services:
  poster-extraction:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Resource Limits

Adjust memory limits based on your system:

```yaml
deploy:
  resources:
    limits:
      memory: 32G
    reservations:
      memory: 16G
```

### Multiple GPUs

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2
          capabilities: [gpu]
```

## Troubleshooting

### GPU Not Detected

```
nvidia-container-cli: initialization error
```

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Install nvidia-container-toolkit:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
3. Restart Docker Desktop (Windows)

### Out of Memory

```
torch.cuda.OutOfMemoryError
```

**Solutions:**
- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Process fewer posters concurrently
- Use smaller batch sizes

### Container Won't Start

```bash
# Check logs
docker compose logs

# Check container status
docker ps -a

# Remove and rebuild
docker compose down
docker compose build --no-cache
docker compose up
```

### Port Already in Use

```
Error: bind: address already in use
```

**Solutions:**
```bash
# Find process using port
lsof -i :8000

# Use different port
API_PORT=8001 docker compose up
```

### Model Download Slow/Fails

**Solutions:**
- Check internet connection
- Pre-download models to cache:
  ```bash
  docker run -it --rm \
    -v model-cache:/root/.cache \
    poster-extraction \
    python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('jimnoneill/Llama-3.1-8B-Poster-Extraction')"
  ```

## Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker compose up` | Start containers |
| `docker compose up -d` | Start in background |
| `docker compose down` | Stop containers |
| `docker compose down -v` | Stop and remove volumes |
| `docker compose logs -f` | Follow logs |
| `docker compose restart` | Restart containers |
| `docker compose build --no-cache` | Rebuild from scratch |
| `docker system prune` | Clean unused resources |

## Next Steps

- [API Reference](API.md) - Using the REST API
- [Architecture](ARCHITECTURE.md) - Technical details
- [Evaluation](EVALUATION.md) - Validation methodology

