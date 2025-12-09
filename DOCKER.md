# Docker Setup

This document describes how to run the poster extraction pipeline using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (≥16GB VRAM recommended)

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

## Building the Image

```bash
docker-compose build
```

Or build directly:

```bash
docker build -t poster-extraction:latest .
```

### Building for Server Deployment

For production server deployment, you may want to build and push to a registry:

```bash
# Build with a specific tag
docker build -t your-registry/poster-extraction:v1.0.0 .

# Push to registry
docker push your-registry/poster-extraction:v1.0.0
```

## Running

### Production Mode

```bash
docker-compose up
```

This will:

- Process all posters in `./manual_poster_annotation/`
- Output results to `./output/`
- Use GPU acceleration automatically

### Development Mode

For development with live code reloading:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This mounts the source code as a volume, so changes to `poster_extraction.py` are reflected immediately (though you may need to restart the container for some changes).

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
2. **Verify GPU Access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
3. **Ensure sufficient resources**:
   - GPU: ≥16GB VRAM
   - RAM: ≥32GB recommended
   - Disk: ≥50GB free (for models and outputs)

### Deployment Steps

1. **Clone repository on server**:

   ```bash
   git clone <repository-url>
   cd posters-science-posterextraction-beta
   ```

2. **Build the image**:

   ```bash
   docker-compose build
   ```

3. **Configure volumes** (update paths in `docker-compose.yml` as needed):

   - Input directory: Where posters are stored
   - Output directory: Where results will be written
   - Model cache: Persistent location for HuggingFace cache

4. **Start the service**:

   ```bash
   docker-compose up -d
   ```

5. **Monitor logs**:

   ```bash
   docker-compose logs -f poster-extraction
   ```

6. **Check container health**:
   ```bash
   docker-compose ps
   docker inspect poster-extraction | grep -A 10 Health
   ```

### Running as a Service

For systemd service integration:

Create `/etc/systemd/system/poster-extraction.service`:

```ini
[Unit]
Description=Poster Extraction Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/posters-science-posterextraction-beta
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl enable poster-extraction.service
sudo systemctl start poster-extraction.service
```

## Configuration

### Environment Variables

You can override environment variables via `.env` file or directly in `docker-compose.yml`:

- `CUDA_VISIBLE_DEVICES`: Which GPU to use (default: `0`)
- `GPU_COUNT`: Number of GPUs to use (default: `1`)
- `RESTART_POLICY`: Container restart policy (default: `unless-stopped`)
- `PDFALTO_PATH`: Path to pdfalto binary (optional, auto-detected if in PATH)

Create a `.env` file in the project root:

```bash
CUDA_VISIBLE_DEVICES=0
GPU_COUNT=1
RESTART_POLICY=unless-stopped
```

### Volumes

The docker-compose setup mounts:

- `./manual_poster_annotation` → `/app/input` (read-only)
- `./output` → `/app/output` (read-write)
- `~/.cache/huggingface` → `/root/.cache/huggingface` (for model cache)

**For server deployment**, update volume paths to absolute paths:

```yaml
volumes:
  - /data/posters/input:/app/input:ro
  - /data/posters/output:/app/output
  - /data/huggingface-cache:/root/.cache/huggingface
```

### Resource Limits

The docker-compose file includes resource limits:

- Memory limit: 32GB
- Memory reservation: 16GB
- GPU: 1 GPU (configurable via `GPU_COUNT`)

Adjust these in `docker-compose.yml` based on your server's capabilities.

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Docker runtime is installed and verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### pdfalto Build Fails

If pdfalto fails to build in the Dockerfile, you can:

1. Use a pre-built binary by setting `PDFALTO_PATH` environment variable
2. Comment out the pdfalto build section and install it manually
3. Use PyMuPDF fallback (the code will automatically fall back if pdfalto is unavailable)

### Out of Memory

If you encounter OOM errors:

- Reduce batch size or process fewer posters at once
- Use a smaller model variant
- Increase Docker memory limits in Docker Desktop settings
- Adjust memory limits in `docker-compose.yml`

## Model Downloads

Models are automatically downloaded on first run and cached in `~/.cache/huggingface`. This volume is mounted to persist the cache between container runs, speeding up subsequent executions.

## Notes

- The first run will take longer as models are downloaded
- Ensure you have sufficient disk space for model cache (~15-20GB)
- GPU memory usage peaks during model loading and inference
- Health checks verify CUDA availability every 30 seconds
