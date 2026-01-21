# API Reference

REST API documentation for poster2json.

## Quick Start

```bash
# Start the API server
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
  "message": "Poster Extraction API is running"
}
```

#### `GET /health`

Detailed health status including GPU information.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_free": "20.5 GB"
}
```

### Extract Poster

#### `POST /extract`

Extract structured JSON from an uploaded poster file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Supported formats:**
- PDF (`.pdf`)
- Images (`.jpg`, `.jpeg`, `.png`)

**Example:**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@poster.pdf"
```

**Success Response (200):**
```json
{
  "$schema": "https://posters.science/schema/v0.1/poster_schema.json",
  "creators": [
    {
      "name": "Smith, John",
      "affiliation": [{"name": "University of Example"}]
    }
  ],
  "titles": [{"title": "Research Poster Title"}],
  "posterContent": {
    "sections": [
      {"sectionTitle": "Abstract", "sectionContent": "..."},
      {"sectionTitle": "Methods", "sectionContent": "..."}
    ]
  },
  "imageCaptions": [
    {"captions": ["Figure 1.", "Description of figure"]}
  ],
  "tableCaptions": {
    "captions": []
  }
}
```

**Error Response (400):**
```json
{
  "error": "No file provided"
}
```

**Error Response (500):**
```json
{
  "error": "Extraction failed: [error details]"
}
```

## Usage Examples

### Python (requests)

```python
import requests

url = "http://localhost:8000/extract"
files = {"file": open("poster.pdf", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(result["titles"][0]["title"])
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/extract', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.titles[0].title);
```

### cURL

```bash
# Extract from PDF
curl -X POST http://localhost:8000/extract \
  -F "file=@poster.pdf" \
  -o result.json

# Extract from image
curl -X POST http://localhost:8000/extract \
  -F "file=@poster.jpg" \
  -o result.json
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API server port | 8000 |
| `HOST` | API server host | 0.0.0.0 |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) | All available |

### Starting with Custom Port

```bash
PORT=9000 python api.py
```

## Error Handling

The API returns appropriate HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (missing file, invalid format) |
| 500 | Server error (extraction failed) |

## Rate Limiting

The API processes one poster at a time due to GPU memory constraints. Concurrent requests are queued.

For high-throughput scenarios:
- Deploy multiple containers
- Use batch processing via CLI instead

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

