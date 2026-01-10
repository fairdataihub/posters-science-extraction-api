# Docker Setup

This document describes how to run the poster extraction pipeline using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (≥24GB VRAM recommended)
- HuggingFace account and access token

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

- Start the Flask API server on port 8000
- Use GPU acceleration automatically
- Load models from HuggingFace (cached after first run)

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
   - GPU: ≥24GB VRAM (for both Llama 3.1 and Qwen2-VL models)
   - RAM: ≥32GB recommended
   - Disk: ≥50GB free (for models and outputs)

### Deployment Steps

1. **Clone repository on server**:

   ```bash
   git clone <repository-url>
   cd posters-science-posterextraction-beta
   ```

2. **Configure environment**:

   Create a `.env` file with your HuggingFace token:
   ```bash
   HF_TOKEN=your_huggingface_token_here
   CUDA_VISIBLE_DEVICES=0
   GPU_COUNT=1
   RESTART_POLICY=unless-stopped
   ```

3. **Build the image**:

   ```bash
   docker-compose build
   ```

4. **Start the service**:

   ```bash
   docker-compose -f docker-compose-prod.yml up -d
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
ExecStart=/usr/bin/docker-compose -f docker-compose-prod.yml up -d
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

- `HF_TOKEN`: HuggingFace access token (required for model download)
- `CUDA_VISIBLE_DEVICES`: Which GPU to use (default: `0`)
- `GPU_COUNT`: Number of GPUs to use (default: `1`)
- `RESTART_POLICY`: Container restart policy (default: `unless-stopped`)
- `PDFALTO_PATH`: Path to pdfalto binary (optional, auto-detected if in PATH)

Create a `.env` file in the project root:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
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

- Memory limit: 48GB
- Memory reservation: 24GB
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

- Process PDFs and images separately (unloads vision model between phases)
- Reduce batch size or process fewer posters at once
- Increase Docker memory limits in Docker Desktop settings
- Adjust memory limits in `docker-compose.yml`

### Model Download Issues

If models fail to download:

1. Verify your HuggingFace token is valid
2. Accept the model license at https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction
3. Check network connectivity
4. Pre-download models by running: `huggingface-cli download jimnoneill/Llama-3.1-8B-Poster-Extraction`

## Model Downloads

Models are automatically downloaded on first run and cached in `~/.cache/huggingface`. This volume is mounted to persist the cache between container runs, speeding up subsequent executions.

Required models:
- `jimnoneill/Llama-3.1-8B-Poster-Extraction` (~16GB)
- `Qwen/Qwen2-VL-7B-Instruct` (~15GB)

## Notes

- The first run will take longer as models are downloaded (~31GB total)
- Ensure you have sufficient disk space for model cache
- GPU memory usage peaks during model loading and inference
- Health checks verify CUDA availability every 30 seconds
- The API includes a `/health` endpoint for load balancer integration
