# API Reference

REST API and job worker for poster extraction.

## Overview

The API **does not accept file uploads**. The frontend uploads poster files to Bunny storage and creates `ExtractionJob` records in the database. This service polls the database for new jobs, downloads the file from Bunny, runs extraction, and writes results to `PosterMetadata`.

## Quick Start

```bash
# Set required environment variables (see Configuration)
export DATABASE_URL="postgresql://..."
export BUNNY_STORAGE_ZONE="your-zone"
export BUNNY_ACCESS_KEY="your-storage-password"

# Start the API server (starts background job worker)
python api.py

# Or via Docker
docker compose up
```

The API runs on `http://localhost:8000` by default.

## Endpoints

### Health Check

#### `GET /`

Simple health check returning API status.

**Response:**
```json
{
  "status": "ok",
  "service": "Poster Extraction API",
  "version": "1.0.0"
}
```

#### `GET /health`  
#### `GET /up`

Detailed health status including GPU and model availability.

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "api": "ok",
    "cuda": "ok",
    "gpu": "NVIDIA GeForce RTX 4090",
    "json_model": "ok"
  }
}
```

### Trigger job check

#### `POST /jobs/check`

Run one job-worker cycle: if there is an uncompleted (pending) job, it is claimed and processed immediately. Call this after submitting a job to start processing without waiting for the next poll interval.

**Response:** `204 No Content` (no body).

## Job Worker

A background thread runs continuously:

1. **Poll** the database for an `ExtractionJob` with `completed = false` and `status = 'pending'`.
2. **Claim** the job (set `status = 'processing'`).
3. **Download** the file from Bunny storage using the job’s `filePath` (and optional `fileName`).
4. **Extract** using the same pipeline as the CLI (no extraction logic changes).
5. **Upsert** `PosterMetadata` for the job’s `posterId` with the extracted JSON (creators, titles, posterContent, imageCaption, tableCaption, etc.).
6. **Complete** the job (`status = 'completed'`, `completed = true`) or **fail** it (`status = 'failed'`, `error` set).

Only one extraction runs at a time (shared lock with any future HTTP-triggered work).

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL (for ExtractionJob / PosterMetadata) | *required* |
| `BUNNY_STORAGE_ZONE` | Bunny storage zone name | *required* |
| `BUNNY_ACCESS_KEY` | Bunny storage zone password (AccessKey) | *required* |
| `BUNNY_REGION` | Optional region (e.g. `ny`, `uk`); omit for default `storage.bunnycdn.com` | — |
| `POLL_INTERVAL_SECONDS` | Seconds between job poll cycles | 30 |
| `PORT` | API server port | 8000 |
| `HOST` | API server host | 0.0.0.0 |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) | All available |
| `PDFALTO_PATH` | Path to pdfalto binary (for PDF processing) | See poster_extraction |

### Starting with Custom Port

```bash
PORT=9000 python api.py
```

## Error Handling

| Code | Description |
|------|-------------|
| 200 | Success (health) |
| 503 | Unhealthy (e.g. GPU or model unavailable) |

Job failures are recorded in the database: `ExtractionJob.status = 'failed'` and `ExtractionJob.error` set.

## CORS

CORS is enabled by default for all origins. Configure in `api.py` if needed:

```python
from flask_cors import CORS
CORS(app, origins=["https://your-domain.com"])
```

## See Also

- [Docker Setup](DOCKER.md) - Container deployment
- [Architecture](ARCHITECTURE.md) - Technical details
- [Installation](INSTALLATION.md) - Setup instructions
