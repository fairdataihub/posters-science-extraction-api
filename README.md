# Posters Science Extraction API

Convert scientific posters (PDF/images) into structured JSON metadata using Large Language Models.

## Quick Start

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# If installed via pip
poster2json --annotation-dir "./posters" --output-dir "./output"

# Or run directly
python poster_extraction.py --annotation-dir "./posters" --output-dir "./output"
```

### Docker (Recommended for Windows)

```bash
docker compose up --build
```

See [Docker Setup](docs/DOCKER.md) for detailed instructions including Windows/WSL2 support.

## How It Works

```
PDF/Image → Raw Text Extraction → LLM JSON Structuring → Structured JSON
                ↓                         ↓
           [pdfalto]              [Llama 3.1 8B]
           [Qwen2-VL]             Fine-tuned for posters
```

1. **PDF files** → Processed via `pdfalto` for layout-aware text extraction
2. **Image files** → Processed via `Qwen2-VL-7B` vision-language model
3. **All files** → Structured into JSON by [Llama-3.1-8B-Poster-Extraction](https://huggingface.co/jimnoneill/Llama-3.1-8B-Poster-Extraction)

## Output Format

Output conforms to the [poster-json-schema](https://github.com/fairdataihub/poster-json-schema):

```json
{
  "$schema": "https://posters.science/schema/v0.1/poster_schema.json",
  "creators": [
    {
      "name": "LastName, FirstName",
      "givenName": "FirstName",
      "familyName": "LastName",
      "affiliation": ["Institution"]
    }
  ],
  "titles": [{ "title": "Poster Title" }],
  "posterContent": {
    "sections": [
      { "sectionTitle": "Abstract", "sectionContent": "..." },
      { "sectionTitle": "Methods", "sectionContent": "..." }
    ]
  },
  "imageCaptions": [{ "captions": ["Figure 1.", "Description"] }],
  "tableCaptions": [{ "captions": ["Table 1.", "Description"] }]
}
```

## System Requirements

| Requirement | Specification                           |
| ----------- | --------------------------------------- |
| GPU         | CUDA-capable, ≥16GB VRAM                |
| RAM         | ≥32GB recommended                       |
| Python      | 3.10+                                   |
| OS          | Linux, macOS, Windows (via Docker/WSL2) |

## API Server

The API does **not** accept file uploads. The frontend uploads poster files to Bunny storage and creates `ExtractionJob` records in the database. This service polls the database for new jobs, downloads the file from Bunny, runs extraction, and writes results to `PosterMetadata`.

```bash
# Set required environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export BUNNY_STORAGE_ZONE="your-storage-zone"
export BUNNY_ACCESS_KEY="your-storage-zone-password"

# Start the API (starts background job worker)
python api.py

# Health check
curl http://localhost:8000/health
```

See [API Reference](docs/API.md) for full configuration and environment variables.

## Documentation

| Document                                   | Description                         |
| ------------------------------------------ | ----------------------------------- |
| [Installation Guide](docs/INSTALLATION.md) | Detailed setup instructions         |
| [Docker Setup](docs/DOCKER.md)             | Docker deployment & Windows support |
| [Architecture](docs/ARCHITECTURE.md)       | Technical details & methodology     |
| [Evaluation](docs/EVALUATION.md)           | Validation metrics & results        |
| [API Reference](docs/API.md)               | REST API documentation              |

## Project Structure

```
poster2json/
├── poster_extraction.py    # Main extraction pipeline
├── api.py                  # Flask REST API
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build
├── docker-compose.yml      # Docker orchestration
├── docs/                   # Documentation
├── example_posters/        # Sample poster files
└── test_results/           # Validation outputs
```

## Performance

**Validation Results**: 10/10 (100%) passing on test set

| Metric           | Score | Threshold |
| ---------------- | ----- | --------- |
| Word Capture     | 0.96  | ≥0.75     |
| ROUGE-L          | 0.89  | ≥0.75     |
| Number Capture   | 0.93  | ≥0.75     |
| Field Proportion | 0.99  | 0.50–2.00 |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

Part of the [FAIR Data Innovations Hub](https://fairdataihub.org/) posters.science project.

## Contributing

Contributions welcome! Please open an issue to discuss proposed changes.
