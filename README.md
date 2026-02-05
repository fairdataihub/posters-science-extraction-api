# poster2json

Convert scientific posters (PDF/images) into structured JSON metadata using Large Language Models.

[![DOI](https://zenodo.org/badge/1105067405.svg)](https://doi.org/10.5281/zenodo.18319796)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

### Installation

**Option 1: pip install from GitHub**
```bash
pip install git+https://github.com/fairdataihub/posters-science-extraction-api.git
```

**Option 2: Clone and install**
```bash
git clone https://github.com/fairdataihub/posters-science-extraction-api.git
cd posters-science-extraction-api
pip install -e .
```

**Option 3: Requirements only**
```bash
git clone https://github.com/fairdataihub/posters-science-extraction-api.git
cd posters-science-extraction-api
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
  "creators": [{"name": "LastName, FirstName", "givenName": "FirstName", "familyName": "LastName", "affiliation": ["Institution"]}],
  "titles": [{"title": "Poster Title"}],
  "posterContent": {
    "sections": [
      {"sectionTitle": "Abstract", "sectionContent": "..."},
      {"sectionTitle": "Methods", "sectionContent": "..."}
    ]
  },
  "imageCaptions": [{"captions": ["Figure 1.", "Description"]}],
  "tableCaptions": [{"captions": ["Table 1.", "Description"]}]
}
```

## System Requirements

| Requirement | Specification |
|-------------|---------------|
| GPU | CUDA-capable, ≥16GB VRAM |
| RAM | ≥32GB recommended |
| Python | 3.10+ |
| OS | Linux, macOS, Windows (via Docker/WSL2) |

## API Server

```bash
# Start the API
python api.py

# POST a poster file
curl -X POST http://localhost:8000/extract -F "file=@poster.pdf"
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Detailed setup instructions |
| [Docker Setup](docs/DOCKER.md) | Docker deployment & Windows support |
| [Architecture](docs/ARCHITECTURE.md) | Technical details & methodology |
| [Evaluation](docs/EVALUATION.md) | Validation metrics & results |
| [API Reference](docs/API.md) | REST API documentation |

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

| Metric | Score | Threshold |
|--------|-------|-----------|
| Word Capture | 0.96 | ≥0.75 |
| ROUGE-L | 0.89 | ≥0.75 |
| Number Capture | 0.93 | ≥0.75 |
| Field Proportion | 0.99 | 0.50–2.00 |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

Part of the [FAIR Data Innovations Hub](https://fairdataihub.org/) posters.science project.

## Contributing

Contributions welcome! Please open an issue to discuss proposed changes.
